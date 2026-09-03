"""Prometheus text exposition for registered eval runs (no prometheus_client)."""

from __future__ import annotations

from datetime import datetime

from uzbek_ner.evaldash.schema import EvalRun
from uzbek_ner.labels import ENTITY_LABELS

_METRIC_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("ner_eval_f1", "f1", "Exact-span F1 (same hash and exact label/start/end)"),
    ("ner_eval_precision", "precision", "Exact-span precision"),
    ("ner_eval_recall", "recall", "Exact-span recall"),
)


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _label_set(run: EvalRun, label: str) -> str:
    pairs = (
        ("run_id", run.run_id),
        ("model", run.model),
        ("split", run.split),
        ("label", label),
    )
    return ",".join(f'{key}="{_escape(val)}"' for key, val in pairs)


def _identity_labels(run: EvalRun) -> str:
    pairs = (
        ("run_id", run.run_id),
        ("model", run.model),
        ("split", run.split),
    )
    return ",".join(f'{key}="{_escape(val)}"' for key, val in pairs)


def created_at_timestamp(run: EvalRun) -> float:
    raw = run.created_at.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(raw).timestamp()
    except ValueError:
        return 0.0


def latest_run(runs: list[EvalRun]) -> EvalRun | None:
    if not runs:
        return None
    return max(runs, key=lambda run: (created_at_timestamp(run), run.run_id))


def render_prometheus(runs: list[EvalRun]) -> str:
    """Gauge text for scrape. Labels: run_id, model, split, label."""

    lines: list[str] = []
    newest = latest_run(runs)
    newest_id = newest.run_id if newest is not None else None

    for metric_name, field, help_text in _METRIC_FIELDS:
        lines.append(f"# HELP {metric_name} {help_text}")
        lines.append(f"# TYPE {metric_name} gauge")
        for run in runs:
            micro = getattr(run.metrics.micro, field)
            lines.append(f"{metric_name}{{{_label_set(run, 'micro')}}} {micro}")
            macro = getattr(run.metrics.macro, field)
            lines.append(f"{metric_name}{{{_label_set(run, 'macro')}}} {macro}")
            for entity in ENTITY_LABELS:
                value = getattr(run.metrics.by_label[entity], field)
                lines.append(f"{metric_name}{{{_label_set(run, entity)}}} {value}")

    lines.append("# HELP ner_eval_records Gold documents in the scored split")
    lines.append("# TYPE ner_eval_records gauge")
    for run in runs:
        lines.append(f"ner_eval_records{{{_identity_labels(run)}}} {run.metrics.records}")

    lines.append("# HELP ner_eval_created_timestamp_seconds Run created_at as Unix seconds")
    lines.append("# TYPE ner_eval_created_timestamp_seconds gauge")
    for run in runs:
        ts = created_at_timestamp(run)
        lines.append(f"ner_eval_created_timestamp_seconds{{{_identity_labels(run)}}} {ts}")

    lines.append("# HELP ner_eval_latest 1 if this is the newest registered run")
    lines.append("# TYPE ner_eval_latest gauge")
    for run in runs:
        flag = 1 if run.run_id == newest_id else 0
        lines.append(f"ner_eval_latest{{{_identity_labels(run)}}} {flag}")

    diag_runs = [run for run in runs if run.diagnostics is not None]
    if diag_runs:
        lines.append(
            "# HELP ner_eval_boundary_f1 Type-agnostic exact (start,end) F1; not the organizer score"
        )
        lines.append("# TYPE ner_eval_boundary_f1 gauge")
        for run in diag_runs:
            assert run.diagnostics is not None
            lines.append(
                f"ner_eval_boundary_f1{{{_identity_labels(run)}}} {run.diagnostics.boundary_exact.f1}"
            )
        lines.append(
            "# HELP ner_eval_type_accuracy Label accuracy on spans with matching character offsets"
        )
        lines.append("# TYPE ner_eval_type_accuracy gauge")
        for run in diag_runs:
            assert run.diagnostics is not None
            acc = run.diagnostics.type_given_boundary.accuracy
            lines.append(f"ner_eval_type_accuracy{{{_identity_labels(run)}}} {acc}")
        lines.append("# HELP ner_eval_error_bucket Diagnostic entity counts (not organizer TP/FP)")
        lines.append("# TYPE ner_eval_error_bucket gauge")
        for run in diag_runs:
            assert run.diagnostics is not None
            buckets = run.diagnostics.buckets
            values = {
                "exact_match": buckets.exact_match,
                "type_mismatch": buckets.type_mismatch,
                "partial_same_type": buckets.partial_same_type,
                "partial_diff_type": buckets.partial_diff_type,
                "missed": buckets.missed,
                "spurious": buckets.spurious,
            }
            for bucket, value in values.items():
                labels = f'{_identity_labels(run)},bucket="{bucket}"'
                lines.append(f"ner_eval_error_bucket{{{labels}}} {value}")

    return "\n".join(lines) + "\n"
