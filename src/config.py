from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = Path("config/business_config.json")


@lru_cache(maxsize=1)
def get_business_config(config_path: str | None = None) -> dict[str, Any]:
    path = Path(config_path or DEFAULT_CONFIG_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_labels() -> list[str]:
    return list(get_business_config()["labels"])


def get_no_finding_label() -> str:
    return str(get_business_config()["no_finding_label"])


def get_thresholds(labels: list[str] | None = None) -> dict[str, float]:
    cfg = get_business_config().get("thresholds", {})
    default_threshold = float(cfg.get("default", 0.5))
    overrides = cfg.get("overrides", {})

    if labels is None:
        result: dict[str, float] = {"default": default_threshold}
        for key, value in overrides.items():
            result[str(key)] = float(value)
        return result

    return {str(label): float(overrides.get(label, default_threshold)) for label in labels}


def get_model_settings() -> dict[str, Any]:
    return dict(get_business_config().get("model_settings", {}))


def get_gradcam_settings() -> dict[str, Any]:
    return dict(get_business_config().get("gradcam_settings", {}))


def get_label_descriptions() -> dict[str, str]:
    return {
        str(k): str(v)
        for k, v in get_business_config().get("gradcam_settings", {}).get("label_descriptions", {}).items()
    }


def get_reporting_messages() -> dict[str, str]:
    return {
        str(k): str(v)
        for k, v in get_business_config().get("reporting_messages", {}).items()
    }


def get_ui_settings() -> dict[str, Any]:
    return dict(get_business_config().get("ui_settings", {}))
