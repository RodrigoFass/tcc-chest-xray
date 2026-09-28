from pathlib import Path

import pytest
import yaml

from chestxray.config import deep_merge, load_config, save_config

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
PATH_FILES = sorted((CONFIGS / "paths").glob("*.yaml"))
EXPERIMENT_FILES = sorted(p for p in CONFIGS.rglob("*.yaml") if p.parent.name != "paths")


def write_yaml(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_deep_merge_merges_nested_and_replaces_lists():
    base = {"a": 1, "nested": {"x": 1, "y": 2}, "items": [1, 2, 3]}
    override = {"nested": {"y": 20, "z": 30}, "items": [9]}
    merged = deep_merge(base, override)
    assert merged == {"a": 1, "nested": {"x": 1, "y": 20, "z": 30}, "items": [9]}
    assert base["nested"] == {"x": 1, "y": 2}  # inputs are not mutated


def test_debug_inherits_from_base():
    cfg = load_config(CONFIGS / "debug.yaml", CONFIGS / "paths" / "local.yaml")
    assert "extends" not in cfg
    assert cfg["experiment"] == "debug"
    assert cfg["train"]["batch_size"] == 8
    assert cfg["train"]["max_epochs"] == 1
    assert cfg["train"]["scheduler"] == {"factor": 0.1, "patience": 1}  # untouched parent block
    assert isinstance(cfg["train"]["lr"], float) and cfg["train"]["lr"] == pytest.approx(1e-4)


def test_base_classes():
    classes = load_config(CONFIGS / "base.yaml")["data"]["classes"]
    assert len(classes) == len(set(classes)) == 14
    assert "Pleural_Thickening" in classes
    assert set(load_config(CONFIGS / "base.yaml")["data"]["focus_classes"]) <= set(classes)


def test_local_paths_resolve_to_repo():
    paths = load_config(CONFIGS / "base.yaml", CONFIGS / "paths" / "local.yaml")["paths"]
    assert set(paths) >= {"data_dir", "runs_dir", "checkpoint_dir", "splits_dir", "results_dir"}
    assert Path(paths["splits_dir"]) == ROOT / "data" / "splits"
    assert Path(paths["runs_dir"]) == ROOT / "results" / "runs"


@pytest.mark.parametrize("config_file", EXPERIMENT_FILES, ids=lambda p: p.stem)
@pytest.mark.parametrize("paths_file", PATH_FILES, ids=lambda p: p.stem)
def test_every_config_loads_with_every_paths_file(config_file, paths_file):
    cfg = load_config(config_file, paths_file)
    assert {"data_dir", "runs_dir", "checkpoint_dir"} <= set(cfg["paths"])


def test_multi_level_extends(tmp_path):
    write_yaml(tmp_path / "a.yaml", {"x": 1, "block": {"p": 1, "q": 1}})
    write_yaml(tmp_path / "sub" / "b.yaml", {"extends": "../a.yaml", "block": {"q": 2}})
    c = write_yaml(tmp_path / "sub" / "c.yaml", {"extends": "b.yaml", "x": 3})
    assert load_config(c) == {"x": 3, "block": {"p": 1, "q": 2}}


def test_circular_extends_raises(tmp_path):
    write_yaml(tmp_path / "a.yaml", {"extends": "b.yaml"})
    write_yaml(tmp_path / "b.yaml", {"extends": "a.yaml"})
    with pytest.raises(ValueError, match="Circular"):
        load_config(tmp_path / "a.yaml")


def test_paths_file_only_accepts_paths(tmp_path):
    cfg = write_yaml(tmp_path / "exp.yaml", {"train": {"batch_size": 32}})
    bad = write_yaml(tmp_path / "env.yaml", {"paths": {"data_dir": "d"}, "train": {"batch_size": 64}})
    with pytest.raises(ValueError, match="may only define"):
        load_config(cfg, bad)


def test_relative_paths_use_project_root_not_cwd(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    cfg = write_yaml(tmp_path / "configs" / "exp.yaml", {"paths": {"data_dir": "data"}})
    monkeypatch.chdir(tmp_path / "configs")  # e.g. a notebook running from notebooks/
    assert Path(load_config(cfg)["paths"]["data_dir"]) == (tmp_path / "data").resolve()


def test_paths_expand_env_vars(tmp_path, monkeypatch):
    monkeypatch.setenv("TCC_TEST_DATA", str(tmp_path / "somewhere"))
    cfg = write_yaml(tmp_path / "exp.yaml", {"paths": {"data_dir": "$TCC_TEST_DATA/nih256"}})
    assert Path(load_config(cfg)["paths"]["data_dir"]) == tmp_path / "somewhere" / "nih256"


def test_save_config_round_trip(tmp_path):
    cfg = load_config(CONFIGS / "debug.yaml", CONFIGS / "paths" / "local.yaml")
    out = tmp_path / "run" / "config.yaml"
    save_config(cfg, out)
    assert yaml.safe_load(out.read_text(encoding="utf-8")) == cfg
