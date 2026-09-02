import uzbek_ner
from uzbek_ner.settings import get_settings
from uzbek_ner.tracking import RESEARCH_EXPERIMENT, SMOKE_EXPERIMENT


def test_version() -> None:
    assert uzbek_ner.__version__ == "0.1.0"


def test_settings_defaults() -> None:
    settings = get_settings()
    assert settings.mlflow_experiment == "uzbek_ner"
    assert settings.data_raw.name == "raw"


def test_tracking_constants() -> None:
    assert RESEARCH_EXPERIMENT == "uzbek_ner"
    assert SMOKE_EXPERIMENT == "uzbek_ner_smoke"
