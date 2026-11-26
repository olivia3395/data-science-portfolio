from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import yaml


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_json(obj: dict[str, Any], path: str | Path) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, default=str)


def dump_joblib(obj: Any, path: str | Path) -> None:
    joblib.dump(obj, path)


def load_joblib(path: str | Path) -> Any:
    return joblib.load(path)
