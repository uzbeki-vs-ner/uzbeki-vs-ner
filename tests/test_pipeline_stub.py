"""Smoke tests for pipeline stubs (no dataset required)."""

from hydra import compose, initialize_config_dir

from uzbek_ner.pipeline.prepare import run_prepare
from uzbek_ner.settings import REPO_ROOT, Settings


def _load_config() -> object:
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base=None):
        return compose(config_name="default")


def test_prepare_stub_writes_manifest(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    settings = Settings(
        data_raw=tmp_path / "raw",
        data_processed=tmp_path / "processed",
        checkpoints=tmp_path / "checkpoints",
        models=tmp_path / "models",
        metrics_path=tmp_path / "metrics.json",
    )
    cfg = _load_config()
    manifest = run_prepare(cfg, settings=settings)
    assert manifest.exists()
    assert manifest.name == "manifest.json"
