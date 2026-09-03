"""Local exact-span eval registry (Grafana / comparison API)."""

from uzbek_ner.evaldash.registry import (
    DEFAULT_RUNS_DIR,
    evaluate_and_register,
    get_run,
    ingest_metrics,
    load_runs,
    resolve_runs_dir,
    write_run,
)
from uzbek_ner.evaldash.schema import EvalRun

__all__ = [
    "DEFAULT_RUNS_DIR",
    "EvalRun",
    "evaluate_and_register",
    "get_run",
    "ingest_metrics",
    "load_runs",
    "resolve_runs_dir",
    "write_run",
]
