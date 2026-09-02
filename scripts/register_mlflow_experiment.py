#!/usr/bin/env python3
"""Register default MLflow experiments on the local sqlite backend."""

import mlflow

from uzbek_ner.settings import get_settings
from uzbek_ner.tracking import RESEARCH_EXPERIMENT, SMOKE_EXPERIMENT


def main() -> None:
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    for name in (RESEARCH_EXPERIMENT, SMOKE_EXPERIMENT, settings.mlflow_experiment):
        mlflow.set_experiment(name)
        print(f"registered experiment: {name} @ {settings.mlflow_tracking_uri}")


if __name__ == "__main__":
    main()
