from trajrisk.config import PROJECT_ROOT, load_config


def test_default_paths_start_at_project_folder(monkeypatch):
    monkeypatch.delenv("TRAJRISK_STORAGE_ROOT", raising=False)
    cfg = load_config("configs/base.yaml")
    assert cfg["paths"]["data_dir"] == PROJECT_ROOT / "data"


def test_storage_root_moves_both_datasets(monkeypatch, tmp_path):
    monkeypatch.setenv("TRAJRISK_STORAGE_ROOT", str(tmp_path))
    assert load_config("configs/base.yaml")["paths"]["data_dir"] == tmp_path / "data"
    ml = load_config("configs/movielens.yaml")
    assert ml["paths"]["data_dir"] == tmp_path / "data" / "movielens"
    assert ml["paths"]["output_dir"] == tmp_path / "outputs" / "movielens"
