from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import SequenceClassifierOutput

from ml.training.train_cls import DistilledStudent
from ml.training.trainer_ext import DistillationTrainer


class ConstantLogitsModel(nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("_logits", logits)

    def forward(self, **_: torch.Tensor) -> SequenceClassifierOutput:
        return SequenceClassifierOutput(logits=self._logits.clone())


def _smoothed_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor, smoothing: float
) -> torch.Tensor:
    logits = logits.to(torch.float32)
    if smoothing <= 0:
        return F.cross_entropy(logits, labels)

    log_probs = F.log_softmax(logits, dim=-1)
    nll = -log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    smooth = -log_probs.mean(dim=-1)
    return ((1 - smoothing) * nll + smoothing * smooth).mean()


def _distillation_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    student_scaled = student_logits.to(torch.float32) / temperature
    teacher_scaled = teacher_logits.to(torch.float32) / temperature
    return F.kl_div(
        F.log_softmax(student_scaled, dim=-1),
        F.softmax(teacher_scaled, dim=-1),
        reduction="batchmean",
    ) * (temperature**2)


def _run_forward(
    *,
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    smoothing: float,
    alpha: float,
    temperature: float,
    teacher_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    teacher_model = (
        ConstantLogitsModel(teacher_logits) if teacher_logits is not None else None
    )
    model = DistilledStudent(
        ConstantLogitsModel(student_logits),
        teacher_model=teacher_model,
        alpha=alpha,
        temperature=temperature,
        label_smoothing_factor=smoothing,
    )
    outputs = model(
        input_ids=torch.zeros((labels.shape[0], 4), dtype=torch.long),
        attention_mask=torch.ones((labels.shape[0], 4), dtype=torch.long),
        labels=labels,
        teacher_input_ids=torch.zeros((labels.shape[0], 4), dtype=torch.long)
        if teacher_model is not None
        else None,
        teacher_attention_mask=torch.ones((labels.shape[0], 4), dtype=torch.long)
        if teacher_model is not None
        else None,
    )
    return outputs.loss


def test_label_smoothing_zero_with_distillation_matches_ce_plus_kl() -> None:
    student_logits = torch.tensor([[2.5, -0.25], [0.2, 1.2]], dtype=torch.float16)
    teacher_logits = torch.tensor([[1.0, 0.5], [-1.0, 2.0]], dtype=torch.float16)
    labels = torch.tensor([0, 1], dtype=torch.long)
    alpha = 0.35
    temperature = 2.5

    loss = _run_forward(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        labels=labels,
        smoothing=0.0,
        alpha=alpha,
        temperature=temperature,
    )

    ce = _smoothed_cross_entropy(student_logits, labels, smoothing=0.0)
    kl = _distillation_kl(student_logits, teacher_logits, temperature)
    expected = (1 - alpha) * ce + alpha * kl

    assert torch.isclose(loss, expected, atol=1e-6)


def test_label_smoothing_with_distillation_keeps_kl_term() -> None:
    student_logits = torch.tensor([[3.0, -0.5], [0.1, 0.9]], dtype=torch.bfloat16)
    teacher_logits = torch.tensor([[0.5, 1.5], [1.75, -0.25]], dtype=torch.bfloat16)
    labels = torch.tensor([0, 1], dtype=torch.long)
    alpha = 0.4
    temperature = 2.0
    smoothing = 0.1

    loss = _run_forward(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        labels=labels,
        smoothing=smoothing,
        alpha=alpha,
        temperature=temperature,
    )

    ce = _smoothed_cross_entropy(student_logits, labels, smoothing=smoothing)
    kl = _distillation_kl(student_logits, teacher_logits, temperature)
    expected = (1 - alpha) * ce + alpha * kl

    assert kl > 0
    assert loss > (1 - alpha) * ce
    assert torch.isclose(loss, expected, atol=1e-6)


def test_label_smoothing_without_distillation_uses_smoothed_ce_only() -> None:
    student_logits = torch.tensor([[1.5, -0.5], [0.25, 1.75]], dtype=torch.float16)
    labels = torch.tensor([0, 1], dtype=torch.long)
    smoothing = 0.1

    loss = _run_forward(
        student_logits=student_logits,
        labels=labels,
        smoothing=smoothing,
        alpha=0.4,
        temperature=2.0,
    )

    expected = _smoothed_cross_entropy(student_logits, labels, smoothing=smoothing)

    assert torch.isclose(loss, expected, atol=1e-6)


def test_kl_term_is_nonzero_whenever_teacher_differs(
    smoothing: float = 0.1,
) -> None:
    student_logits = torch.tensor([[1.0, 0.0], [0.5, 1.5]], dtype=torch.float16)
    teacher_logits = torch.tensor([[0.0, 1.0], [1.5, 0.5]], dtype=torch.float16)
    labels = torch.tensor([0, 1], dtype=torch.long)
    alpha = 0.25
    temperature = 1.5

    for smoothing in (0.0, 0.1):
        loss = _run_forward(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=labels,
            smoothing=smoothing,
            alpha=alpha,
            temperature=temperature,
        )
        ce = _smoothed_cross_entropy(student_logits, labels, smoothing=smoothing)
        kl = _distillation_kl(student_logits, teacher_logits, temperature)

        assert kl > 0
        assert loss > (1 - alpha) * ce


def test_distillation_trainer_compute_loss_uses_model_loss() -> None:
    trainer = object.__new__(DistillationTrainer)
    expected_loss = torch.tensor(3.5, dtype=torch.float32)
    expected_logits = torch.tensor([[0.0, 1.0]], dtype=torch.float32)

    class DummyModel(nn.Module):
        def forward(self, **_: torch.Tensor) -> SequenceClassifierOutput:
            return SequenceClassifierOutput(loss=expected_loss, logits=expected_logits)

    inputs = {"input_ids": torch.tensor([[1, 2]]), "labels": torch.tensor([1])}

    loss = DistillationTrainer.compute_loss(trainer, DummyModel(), inputs)
    loss_with_outputs, outputs = DistillationTrainer.compute_loss(
        trainer, DummyModel(), inputs, return_outputs=True
    )

    assert loss is expected_loss
    assert loss_with_outputs is expected_loss
    assert torch.equal(outputs.logits, expected_logits)
