"""CPU tests for the token MLP head (no pretrained weights)."""

import torch

from uzbek_ner.modeling.heads import TokenMLPHead, apply_token_head


class _FakeConfig:
    hidden_size = 8
    hidden_dropout_prob = 0.0
    classifier_dropout = 0.0
    num_labels = 7


class _FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = _FakeConfig()
        self.num_labels = 7
        self.classifier = torch.nn.Linear(8, 7)


def test_mlp_head_maps_tokens_to_bio_logits() -> None:
    head = TokenMLPHead(in_size=8, num_labels=7, hidden_size=8, dropout=0.0)
    logits = head(torch.zeros(2, 5, 8))
    assert logits.shape == (2, 5, 7)


def test_apply_token_head_replaces_linear_with_mlp() -> None:
    model = _FakeModel()
    linear_params = sum(param.numel() for param in model.classifier.parameters())
    apply_token_head(model, kind="mlp", mlp_hidden=8)
    mlp_params = sum(param.numel() for param in model.classifier.parameters())
    assert isinstance(model.classifier, TokenMLPHead)
    assert model.config.ner_head == "mlp"
    assert mlp_params > linear_params
    assert model.classifier(torch.zeros(1, 3, 8)).shape == (1, 3, 7)
