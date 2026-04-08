from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont, ImageOps

from src.model_stub import RealCNNModel
from src.schema import PredictionItem, RegionAttribution


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._forward_handle = target_layer.register_forward_hook(self._forward_hook)
        self._backward_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, _module: Any, _inputs: Any, outputs: Any) -> None:
        self.activations = outputs.detach()

    def _backward_hook(
        self,
        _module: Any,
        _grad_inputs: Any,
        grad_outputs: Any,
    ) -> None:
        self.gradients = grad_outputs[0].detach()

    def close(self) -> None:
        self._forward_handle.remove()
        self._backward_handle.remove()

    def compute(self, inputs: torch.Tensor, class_index: int) -> torch.Tensor:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(inputs)
        score = logits[:, class_index].sum()
        score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations or gradients.")

        pooled_gradients = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (self.activations * pooled_gradients).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=inputs.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cam -= cam.min()
        max_value = cam.max()
        if max_value > 0:
            cam /= max_value
        return torch.from_numpy(cam)


def generate_gradcam_regions(
    model: RealCNNModel,
    image_path: str,
    predictions: list[PredictionItem],
    output_dir: str | Path = Path("outputs/gradcam"),
) -> list[RegionAttribution]:
    positive = [item for item in predictions if item.selected and item.label != "No finding"]
    regions: list[RegionAttribution] = []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not positive:
        positive = sorted(predictions, key=lambda item: item.probability, reverse=True)[:1]

    image_tensor = model.preprocess(image_path)
    base_image = Image.open(image_path).convert("RGB")
    gradcam = GradCAM(model.network, model.target_layer)

    style_tag = "jetgrid"

    for item in positive[:3]:
        class_index = model.labels.index(item.label)
        heatmap = gradcam.compute(image_tensor, class_index)
        description, laterality, lung_zone, coordinates, heatmap_path = _build_region_metadata(
            base_image,
            heatmap,
            output_dir / f"{Path(image_path).stem}_{item.label.replace(' ', '_')}_{style_tag}.png",
            item.label,
        )
        regions.append(
            RegionAttribution(
                label=item.label,
                laterality=laterality,
                lung_zone=lung_zone,
                description=description,
                heatmap_path=str(heatmap_path),
                coordinates=coordinates,
            )
        )

    if not regions:
        regions.append(
            RegionAttribution(
                label="No finding",
                description="No dominant focal abnormality detected on the provided image.",
                heatmap_path=str(output_dir / f"{Path(image_path).stem}_no_finding_{style_tag}.png"),
                coordinates=[0.0, 0.0, 1.0, 1.0],
            )
        )

    gradcam.close()
    return regions


def build_gradcam_montage(
    image_path: str,
    regions: list[RegionAttribution],
    output_path: str | Path | None = None,
    max_rows: int = 3,
) -> Path | None:
    rows: list[tuple[str, Image.Image, Path]] = []
    for region in regions[:max_rows]:
        if not region.heatmap_path:
            continue
        panel_path = Path(region.heatmap_path)
        if not panel_path.exists():
            continue
        rows.append((region.label, Image.open(panel_path).convert("RGB"), panel_path))

    if not rows:
        return None

    max_width = max(image.width for _, image, _ in rows)
    title_height = 34
    row_gap = 16
    padding = 18
    canvas_height = (
        padding * 2
        + sum(title_height + image.height for _, image, _ in rows)
        + row_gap * max(0, len(rows) - 1)
    )
    canvas = Image.new("RGB", (max_width + padding * 2, canvas_height), color=(242, 242, 242))
    draw = ImageDraw.Draw(canvas)

    try:
        title_font = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        title_font = ImageFont.load_default()

    y_cursor = padding
    for index, (label, image, _path) in enumerate(rows):
        row_tag = chr(ord("A") + index)
        draw.text((padding, y_cursor), f"({row_tag}) {label}", fill=(15, 15, 15), font=title_font)
        y_cursor += title_height
        x_offset = padding + (max_width - image.width) // 2
        canvas.paste(image, (x_offset, y_cursor))
        y_cursor += image.height + row_gap

    if output_path is None:
        output_dir = rows[0][2].parent
        output_path = output_dir / f"{Path(image_path).stem}_gradcam_montage.png"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def _build_region_metadata(
    base_image: Image.Image,
    heatmap: torch.Tensor,
    output_path: Path,
    label: str,
) -> tuple[str, str | None, str | None, list[float], Path]:
    heatmap_array = heatmap.numpy()
    heatmap_image = Image.fromarray(np.uint8(heatmap_array * 255), mode="L")
    heatmap_image = heatmap_image.resize(base_image.size)
    grayscale_base = ImageOps.grayscale(base_image).convert("RGB")
    heatmap_color = Image.fromarray(_apply_jet_colormap(heatmap_array), mode="RGB").resize(base_image.size)
    blended = Image.blend(grayscale_base, heatmap_color, alpha=0.46)

    panel = _compose_gradcam_panel(grayscale_base, heatmap_color, blended)
    panel.save(output_path)

    coordinates = _heatmap_bbox(heatmap_array)
    laterality = _infer_laterality(coordinates)
    lung_zone = _infer_lung_zone(coordinates)
    description = _label_to_description(label, laterality, lung_zone)
    return description, laterality, lung_zone, coordinates, output_path


def _heatmap_bbox(heatmap: np.ndarray, threshold: float = 0.45) -> list[float]:
    mask = heatmap >= threshold
    if not mask.any():
        return [0.0, 0.0, 1.0, 1.0]

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    height, width = heatmap.shape
    x1 = float(cols[0] / width)
    y1 = float(rows[0] / height)
    x2 = float((cols[-1] + 1) / width)
    y2 = float((rows[-1] + 1) / height)
    return [max(0.0, x1), max(0.0, y1), min(1.0, x2), min(1.0, y2)]


def _infer_laterality(coordinates: list[float]) -> str | None:
    x1, _, x2, _ = coordinates
    midpoint = (x1 + x2) / 2
    if midpoint < 0.42:
        return "left"
    if midpoint > 0.58:
        return "right"
    return None


def _infer_lung_zone(coordinates: list[float]) -> str | None:
    _, y1, _, y2 = coordinates
    midpoint = (y1 + y2) / 2
    if midpoint < 0.33:
        return "upper"
    if midpoint > 0.66:
        return "lower"
    return "mid"


def _label_to_description(label: str, laterality: str | None, lung_zone: str | None) -> str:
    mapping = {
        "Atelectasis": "linear or plate-like basal opacity consistent with volume loss",
        "Consolidation": "focal air-space opacity suspicious for consolidation",
        "Infiltration": "patchy interstitial or air-space opacity",
        "Pneumothorax": "visible pleural line with absent peripheral lung markings",
        "Edema": "bilateral perihilar interstitial opacity suggestive of edema",
        "Emphysema": "hyperinflation with increased lucency",
        "Fibrosis": "chronic reticular scarring pattern",
        "Effusion": "blunting of the costophrenic angle with dependent opacity",
        "Pneumonia": "lobar or segmental opacity concerning for infection",
        "Pleural_thickening": "pleural-based linear thickening",
        "Cardiomegaly": "enlarged cardiomediastinal silhouette",
        "Mass": "rounded focal opacity requiring correlation",
        "Nodule": "small rounded focal opacity requiring correlation",
        "Hernia": "abnormal mediastinal or diaphragmatic contour",
    }

    base = mapping.get(label, "salient radiographic abnormality")
    location_parts = [part for part in [laterality, lung_zone] if part]
    if location_parts:
        return f"{', '.join(location_parts)} {base}"
    return base


def _compose_gradcam_panel(
    original: Image.Image,
    heatmap: Image.Image,
    overlay: Image.Image,
) -> Image.Image:
    # Keep panel tiles compact so the report stays readable in Streamlit.
    tile_original = _resize_for_panel(original, max_side=250)
    tile_heatmap = _resize_for_panel(heatmap, max_side=250)
    tile_overlay = _resize_for_panel(overlay, max_side=250)

    tiles = [tile_original, tile_heatmap, tile_overlay]
    tile_width = max(tile.width for tile in tiles)
    tile_height = max(tile.height for tile in tiles)

    gap = 8
    padding = 8
    label_height = 18
    canvas_width = padding * 2 + (tile_width * 3) + (gap * 2)
    canvas_height = padding * 2 + label_height + tile_height
    canvas = Image.new("RGB", (canvas_width, canvas_height), color=(248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    try:
        label_font = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        label_font = ImageFont.load_default()

    labels = ["Original", "Heatmap", "Superimposed"]

    x_cursor = padding
    for index, tile in enumerate(tiles):
        x_offset = x_cursor + (tile_width - tile.width) // 2
        y_offset = padding + label_height + (tile_height - tile.height) // 2
        label_text = labels[index]
        label_width = draw.textlength(label_text, font=label_font)
        label_x = x_cursor + (tile_width - int(label_width)) // 2
        draw.text((label_x, padding), label_text, fill=(30, 30, 30), font=label_font)
        canvas.paste(tile, (x_offset, y_offset))
        x_cursor += tile_width + gap

    return canvas


def _resize_for_panel(image: Image.Image, max_side: int) -> Image.Image:
    width, height = image.size
    if width <= 0 or height <= 0:
        return image
    scale = min(max_side / float(width), max_side / float(height), 1.0)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(new_size, Image.Resampling.BICUBIC)


def _apply_jet_colormap(heatmap: np.ndarray) -> np.ndarray:
    values = np.clip(heatmap, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * values - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * values - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * values - 1.0), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return np.uint8(rgb * 255.0)
