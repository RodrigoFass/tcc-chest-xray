"""Load YAML configs with inheritance.

A config file may declare ``extends: <file>`` (relative to its own directory) to inherit
from another config. The parent is loaded first and the child is merged on top of it:
nested mappings are merged key by key; any other value, lists included, replaces the
parent's value.

Machine-specific locations live in a separate paths file (``configs/paths/<env>.yaml``)
that may only contain a ``paths`` section. It is merged last, so the same experiment
config runs unchanged on a local machine, Colab or Kaggle.

Every entry under ``paths`` is expanded (``~`` and ``$VAR``) and, if relative, resolved
against the project root (the nearest directory above the config file that contains
``pyproject.toml``). This keeps paths valid no matter where the command or notebook runs.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

EXTENDS_KEY = "extends"
PATHS_KEY = "paths"


def deep_merge(base: dict, override: dict) -> dict:
    """Return a new dict with ``override`` merged recursively into ``base``."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _read_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping, got {type(data).__name__}")
    return data


def _load_with_inheritance(path: Path, chain: tuple[Path, ...] = ()) -> dict:
    path = path.resolve()
    if path in chain:
        cycle = " -> ".join(p.name for p in (*chain, path))
        raise ValueError(f"Circular '{EXTENDS_KEY}' chain: {cycle}")
    data = _read_yaml(path)
    parent = data.pop(EXTENDS_KEY, None)
    if parent is None:
        return data
    return deep_merge(_load_with_inheritance(path.parent / parent, (*chain, path)), data)


def find_project_root(start: Path) -> Path:
    """Nearest directory at or above ``start`` containing ``pyproject.toml`` (else the cwd)."""
    start = start.resolve()
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd()


def _resolve_paths(paths: dict[str, Any], root: Path) -> dict[str, Any]:
    resolved = {}
    for key, value in paths.items():
        if value is None:
            resolved[key] = None
            continue
        path = Path(os.path.expandvars(os.path.expanduser(str(value))))
        resolved[key] = str(path if path.is_absolute() else (root / path).resolve())
    return resolved


def load_config(config_path: str | Path, paths_path: str | Path | None = None) -> dict:
    """Load an experiment config (following ``extends``) and merge the paths file on top."""
    config_path = Path(config_path)
    cfg = _load_with_inheritance(config_path)

    if paths_path is not None:
        paths_cfg = _load_with_inheritance(Path(paths_path))
        extra = set(paths_cfg) - {PATHS_KEY}
        if extra:
            raise ValueError(
                f"{paths_path}: a paths file may only define '{PATHS_KEY}', "
                f"found {sorted(extra)}; put other settings in the experiment config"
            )
        cfg = deep_merge(cfg, paths_cfg)

    if cfg.get(PATHS_KEY):
        cfg[PATHS_KEY] = _resolve_paths(cfg[PATHS_KEY], find_project_root(config_path.parent))
    return cfg


def save_config(cfg: dict, path: str | Path) -> None:
    """Write a resolved config to YAML, keeping the key order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
