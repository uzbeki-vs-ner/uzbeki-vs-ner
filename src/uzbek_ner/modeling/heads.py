"""Token-level NER heads (linear vs 1-hidden MLP).

``RobertaForTokenClassification`` already does Dropout then ``self.classifier``.
The default classifier is ``nn.Linear(H, 7)``. An MLP head replaces only that
module: ``Linear(H, H) → GELU → Dropout → Linear(H, 7)``. The encoder is unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForTokenClassification
from transformers.modeling_utils import PreTrainedModel

HEAD_SPEC_NAME = "head.json"


class TokenMLPHead(nn.Module):
    """One hidden layer on each token vector, then 7 BIO logits."""

    def __init__(self, in_size: int, num_labels: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.dense = nn.Linear(in_size, hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = self.dense(hidden)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        return self.out_proj(hidden)


def _dropout_prob(config: Any) -> float:
    value = getattr(config, "classifier_dropout", None)
    if value is None:
        value = getattr(config, "hidden_dropout_prob", 0.1)
    if value is None:
        return 0.1
    return float(value)


def apply_token_head(
    model: PreTrainedModel,
    *,
    kind: str,
    mlp_hidden: int | None = None,
) -> PreTrainedModel:
    """Replace ``model.classifier`` in place. ``kind='linear'`` is a no-op."""

    kind = kind.strip().lower()
    hidden_size = int(model.config.hidden_size)
    num_labels = int(model.num_labels)
    if kind == "linear":
        model.config.ner_head = "linear"
        return model
    if kind != "mlp":
        raise ValueError(f"unknown token head {kind!r}")
    width = int(mlp_hidden) if mlp_hidden is not None else hidden_size
    model.classifier = TokenMLPHead(
        hidden_size,
        num_labels,
        hidden_size=width,
        dropout=_dropout_prob(model.config),
    )
    model.config.ner_head = "mlp"
    model.config.mlp_hidden = width
    return model


def read_head_spec(path: Path) -> tuple[str, int | None]:
    spec_path = path / HEAD_SPEC_NAME
    if spec_path.is_file():
        payload = json.loads(spec_path.read_text(encoding="utf-8"))
        hidden = payload.get("mlp_hidden")
        return str(payload.get("ner_head", "linear")), int(hidden) if hidden is not None else None
    config_path = path / "config.json"
    if config_path.is_file():
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        hidden = payload.get("mlp_hidden")
        return str(payload.get("ner_head", "linear")), int(hidden) if hidden is not None else None
    return "linear", None


def write_head_spec(path: Path, *, kind: str, mlp_hidden: int | None) -> None:
    payload = {"ner_head": kind, "mlp_hidden": mlp_hidden}
    (path / HEAD_SPEC_NAME).write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def _load_weight_file(path: Path) -> dict[str, torch.Tensor]:
    safetensors = path / "model.safetensors"
    if safetensors.is_file():
        from safetensors.torch import load_file

        return load_file(str(safetensors))
    bin_path = path / "pytorch_model.bin"
    if bin_path.is_file():
        payload = torch.load(bin_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise RuntimeError(f"unexpected checkpoint payload in {bin_path}")
        return payload
    raise FileNotFoundError(f"no model.safetensors or pytorch_model.bin in {path}")


def load_token_classifier(
    path: Path,
    *,
    num_labels: int | None = None,
    id2label: dict[int, str] | None = None,
    label2id: dict[str, int] | None = None,
    head: str | None = None,
    mlp_hidden: int | None = None,
) -> PreTrainedModel:
    """Load a linear or MLP token-class head without flattening MLP weights."""

    path = path.expanduser().resolve()
    stored_kind, stored_hidden = read_head_spec(path)
    kind = (head or stored_kind).strip().lower()
    hidden = mlp_hidden if mlp_hidden is not None else stored_hidden
    extra: dict[str, Any] = {}
    if num_labels is not None:
        extra["num_labels"] = num_labels
    if id2label is not None:
        extra["id2label"] = id2label
    if label2id is not None:
        extra["label2id"] = label2id

    if kind == "linear":
        return AutoModelForTokenClassification.from_pretrained(
            str(path),
            local_files_only=True,
            **extra,
        )

    if extra:
        model = AutoModelForTokenClassification.from_pretrained(
            str(path),
            local_files_only=True,
            ignore_mismatched_sizes=True,
            **extra,
        )
        if kind == "mlp":
            apply_token_head(model, kind="mlp", mlp_hidden=hidden)
            if stored_kind == "mlp":
                model.load_state_dict(_load_weight_file(path), strict=True)
        return model

    config = AutoConfig.from_pretrained(str(path), local_files_only=True)
    model = AutoModelForTokenClassification.from_config(config)
    apply_token_head(model, kind="mlp", mlp_hidden=hidden)
    state = _load_weight_file(path)
    model.load_state_dict(state, strict=True)
    return model
