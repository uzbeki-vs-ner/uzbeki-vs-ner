"""Centralized runtime settings (env + defaults)."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    hf_token: str | None = Field(default=None, alias="HF_TOKEN")
    cuda_visible_devices: str | None = Field(default=None, alias="CUDA_VISIBLE_DEVICES")

    mlflow_tracking_uri: str = Field(
        default="sqlite:///mlflow.db",
        alias="MLFLOW_TRACKING_URI",
    )
    mlflow_experiment: str = Field(default="uzbek_ner", alias="MLFLOW_EXPERIMENT")
    mlflow_smoke_experiment: str = Field(
        default="uzbek_ner_smoke",
        alias="MLFLOW_SMOKE_EXPERIMENT",
    )

    data_raw: Path = REPO_ROOT / "data" / "raw"
    data_processed: Path = REPO_ROOT / "data" / "processed"
    checkpoints: Path = REPO_ROOT / "checkpoints"
    models: Path = REPO_ROOT / "models"
    metrics_path: Path = REPO_ROOT / "metrics.json"


def get_settings() -> Settings:
    return Settings()
