from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback is not used here, but kept safe.
    tomllib = None  # type: ignore[assignment]


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_runtime_secret(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value:
        return value.strip()

    secrets = _load_local_secrets()
    if name in secrets and str(secrets[name]).strip():
        return str(secrets[name]).strip()

    return default


@lru_cache(maxsize=1)
def _load_local_secrets() -> dict[str, str]:
    secrets: dict[str, str] = {}
    secrets.update(_load_dotenv_file(PROJECT_ROOT / ".env"))
    secrets.update(_load_toml_secrets(PROJECT_ROOT / ".streamlit/secrets.toml"))
    return secrets


def _load_dotenv_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    parsed: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value:
            parsed[key] = value
    return parsed


def _load_toml_secrets(path: Path) -> dict[str, str]:
    if tomllib is None or not path.exists():
        return {}

    with path.open("rb") as handle:
        data = tomllib.load(handle)

    parsed: dict[str, str] = {}
    for key, value in data.items():
        if isinstance(value, (str, int, float, bool)):
            parsed[str(key)] = str(value)
    return parsed