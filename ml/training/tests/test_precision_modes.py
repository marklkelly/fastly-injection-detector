from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput

from ml.training.config_runtime import ConfigError, _normalize_mixed_precision_override
from ml.training.train_cls import DistilledStudent


class ConstantLogitsModel(nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("_logits", logits)

    def forward(self, **_: torch.Tensor) -> SequenceClassifierOutput:
        return SequenceClassifierOutput(logits=self._logits.clone())


@pytest.mark.parametrize("input_dtype", [torch.float16, torch.bfloat16])
def test_distillation_fp32_cast_happens_before_temperature_scaling(
    monkeypatch: pytest.MonkeyPatch, input_dtype: torch.dtype
) -> None:
    """The float32 cast must occur before temperature scaling, not after.

    We spy on F.log_softmax and F.softmax — which receive the post-division
    tensors (logits / T) — to confirm those are already float32 when they
    arrive, proving the cast happened before division.
    """
    import torch.nn.functional as real_F

    real_log_softmax = real_F.log_softmax
    real_softmax = real_F.softmax

    student_logits = torch.tensor([[2.0, -1.0], [-0.5, 1.5]], dtype=input_dtype)
    teacher_logits = torch.tensor([[1.0, 0.5], [0.25, -0.75]], dtype=input_dtype)

    model = DistilledStudent(
        ConstantLogitsModel(student_logits),
        alpha=0.4,
        temperature=2.0,
    )
    model._supervised_loss = lambda logits, labels: torch.tensor(1.0)

    captured: dict[str, torch.dtype] = {}

    def spy_log_softmax(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
        captured["scaled_student_dtype"] = input.dtype
        return real_log_softmax(input, dim=dim)

    def spy_softmax(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
        captured["scaled_teacher_dtype"] = input.dtype
        return real_softmax(input, dim=dim)

    monkeypatch.setattr("ml.training.train_cls.F.log_softmax", spy_log_softmax)
    monkeypatch.setattr("ml.training.train_cls.F.softmax", spy_softmax)

    outputs = model(
        input_ids=torch.zeros((2, 4), dtype=torch.long),
        attention_mask=torch.ones((2, 4), dtype=torch.long),
        labels=torch.tensor([0, 1], dtype=torch.long),
        teacher_logits=teacher_logits,
    )

    # The tensors passed to softmax are (logits / T). They must be float32,
    # which is only possible if the cast to float32 happened before division.
    assert captured["scaled_student_dtype"] == torch.float32, (
        "student logits were not cast to float32 before temperature scaling"
    )
    assert captured["scaled_teacher_dtype"] == torch.float32, (
        "teacher logits were not cast to float32 before temperature scaling"
    )
    assert outputs.loss.dtype == torch.float32


@pytest.mark.parametrize(
    ("cli_values", "expected"),
    [
        ({"bf16": True}, "bf16"),
        ({"fp16": True}, "fp16"),
        ({"mixed_precision": "bf16", "bf16": True}, "bf16"),
    ],
)
def test_normalize_mixed_precision_override_resolves_aliases(
    cli_values: dict[str, object], expected: str
) -> None:
    assert _normalize_mixed_precision_override(cli_values) == expected


def test_normalize_mixed_precision_override_rejects_conflicting_alias() -> None:
    with pytest.raises(ConfigError, match="Conflicting mixed precision CLI overrides"):
        _normalize_mixed_precision_override(
            {"mixed_precision": "fp16", "bf16": True}
        )
