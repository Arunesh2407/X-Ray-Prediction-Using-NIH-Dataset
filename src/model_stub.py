from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from dataclasses import dataclass

from src.schema import LABELS, PredictionItem


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class ChestXrayCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.ModuleList(
            [
                ConvBlock(3, 32),
                ConvBlock(32, 64),
                ConvBlock(64, 128),
                ConvBlock(128, 256),
                ConvBlock(256, 512),
            ]
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Identity(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs
        for block in self.features:
            outputs = block(outputs)
        outputs = self.pool(outputs)
        outputs = torch.flatten(outputs, 1)
        return self.classifier(outputs)


@dataclass
class RealCNNModel:
    weights_path: str
    labels: list[str] = None
    image_size: int = 224
    device: str = "cpu"
    thresholds: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if self.labels is None:
            self.labels = LABELS
        self.weights_path = str(Path(self.weights_path))
        self._device = torch.device(self.device)
        self._model = ChestXrayCNN(num_classes=len(self.labels)).to(self._device)
        self._load_weights(Path(self.weights_path))
        self._model.eval()
        self._preprocess = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        if self.thresholds is None:
            self.thresholds = {label: 0.5 for label in self.labels}
            if "Pneumothorax" in self.thresholds:
                self.thresholds["Pneumothorax"] = 0.35

    def _load_weights(self, weight_path: Path) -> None:
        if not weight_path.exists():
            raise FileNotFoundError(f"Weights file not found: {weight_path}")

        state_dict = torch.load(weight_path, map_location=self._device)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if not isinstance(state_dict, dict):
            raise ValueError("Expected a PyTorch state_dict in the uploaded .pth file.")

        normalized_state_dict: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            normalized_key = key.removeprefix("module.")
            normalized_state_dict[normalized_key] = value

        missing_keys, unexpected_keys = self._model.load_state_dict(normalized_state_dict, strict=False)
        if missing_keys or unexpected_keys:
            raise ValueError(
                "Uploaded weights do not match the fallback CNN architecture. "
                f"Missing keys: {missing_keys}. Unexpected keys: {unexpected_keys}."
            )

    def predict(self, image_path: str) -> list[PredictionItem]:
        image = Image.open(image_path).convert("RGB")
        inputs = self._preprocess(image).unsqueeze(0).to(self._device)

        with torch.no_grad():
            logits = self._model(inputs)
            probabilities = torch.sigmoid(logits).squeeze(0).cpu().tolist()

        if len(probabilities) != len(self.labels):
            raise ValueError(
                f"Model output size {len(probabilities)} does not match label count {len(self.labels)}."
            )

        items: list[PredictionItem] = []
        for label, probability in zip(self.labels, probabilities, strict=True):
            threshold = self.thresholds.get(label, 0.5) if self.thresholds else 0.5
            items.append(
                PredictionItem(
                    label=label,
                    probability=float(probability),
                    threshold=float(threshold),
                    selected=float(probability) >= threshold,
                )
            )
        return items

    def preprocess(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        return self._preprocess(image).unsqueeze(0).to(self._device)

    @property
    def network(self) -> nn.Module:
        return self._model

    @property
    def target_layer(self) -> nn.Module:
        return self._model.features[-1].block[3]


MockCNNModel = RealCNNModel
