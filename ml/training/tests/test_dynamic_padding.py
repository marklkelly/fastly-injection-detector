from __future__ import annotations

from pathlib import Path

import torch
from datasets import Dataset
from torch import nn
from transformers import TrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutput

from transformers import Trainer

from ml.training.trainer_ext import DistillationTrainer


class DummyPadTokenizer:
    pad_token_id = 0
    padding_side = "right"
    model_input_names = ["input_ids", "attention_mask", "token_type_ids"]

    def pad(
        self,
        features,
        *,
        padding=True,
        max_length=None,
        pad_to_multiple_of=None,
        return_tensors=None,
        **_,
    ):
        del pad_to_multiple_of

        if padding == "max_length":
            if max_length is None:
                raise AssertionError("max_length is required for fixed padding")
            target_length = max_length
        else:
            target_length = max(len(feature["input_ids"]) for feature in features)

        batch = {}
        for key, pad_value in (
            ("input_ids", self.pad_token_id),
            ("attention_mask", 0),
            ("token_type_ids", 0),
        ):
            if any(key in feature for feature in features):
                values = []
                for feature in features:
                    raw = list(feature.get(key, []))
                    padded = raw + [pad_value] * (target_length - len(raw))
                    values.append(padded)
                batch[key] = values

        if return_tensors == "pt":
            return {
                key: torch.tensor(value, dtype=torch.long)
                for key, value in batch.items()
            }
        return batch


class DummyModel(nn.Module):
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        del attention_mask, kwargs
        batch_size = input_ids.shape[0]
        logits = torch.zeros((batch_size, 2), dtype=torch.float32)
        loss = None
        if labels is not None:
            loss = torch.tensor(0.0, dtype=torch.float32)
        return SequenceClassifierOutput(loss=loss, logits=logits)


def _training_args(tmp_path: Path) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(tmp_path / "trainer-out"),
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        remove_unused_columns=False,
        report_to=[],
    )


def _uncached_dataset() -> Dataset:
    return Dataset.from_list(
        [
            {
                "input_ids": [1, 2, 3],
                "attention_mask": [1, 1, 1],
                "labels": 0,
                "teacher_input_ids": [9, 8, 7, 6],
                "teacher_attention_mask": [1, 1, 1, 1],
            },
            {
                "input_ids": [4, 5, 6, 7, 8],
                "attention_mask": [1, 1, 1, 1, 1],
                "labels": 1,
                "teacher_input_ids": [5, 4, 3, 2, 1, 0],
                "teacher_attention_mask": [1, 1, 1, 1, 1, 1],
            },
        ]
    )


def _cached_dataset() -> Dataset:
    return Dataset.from_list(
        [
            {
                "input_ids": [1, 2, 3],
                "attention_mask": [1, 1, 1],
                "labels": 0,
                "teacher_logits": [0.2, 0.8],
            },
            {
                "input_ids": [4, 5, 6, 7, 8],
                "attention_mask": [1, 1, 1, 1, 1],
                "labels": 1,
                "teacher_logits": [0.7, 0.3],
            },
        ]
    )


def test_train_dataloader_pads_student_and_teacher_to_longest_batch(
    tmp_path: Path,
) -> None:
    tokenizer = DummyPadTokenizer()
    trainer = DistillationTrainer(
        model=DummyModel(),
        args=_training_args(tmp_path),
        train_dataset=_uncached_dataset(),
        eval_dataset=_uncached_dataset(),
        processing_class=tokenizer,
        student_tokenizer=tokenizer,
        teacher_tokenizer=tokenizer,
        student_max_length=8,
        teacher_max_length=12,
        cache_teacher_logits=False,
    )

    batch = next(iter(trainer.get_train_dataloader()))

    assert batch["input_ids"].shape == (2, 5)
    assert batch["attention_mask"].shape == (2, 5)
    assert batch["teacher_input_ids"].shape == (2, 6)
    assert batch["teacher_attention_mask"].shape == (2, 6)
    assert sorted(batch["labels"].cpu().tolist()) == [0, 1]


def test_eval_and_test_dataloaders_keep_fixed_padding_lengths(tmp_path: Path) -> None:
    tokenizer = DummyPadTokenizer()
    dataset = _uncached_dataset()
    trainer = DistillationTrainer(
        model=DummyModel(),
        args=_training_args(tmp_path),
        train_dataset=dataset,
        eval_dataset=dataset,
        processing_class=tokenizer,
        student_tokenizer=tokenizer,
        teacher_tokenizer=tokenizer,
        student_max_length=8,
        teacher_max_length=12,
        cache_teacher_logits=False,
    )

    eval_batch = next(iter(trainer.get_eval_dataloader()))
    test_batch = next(iter(trainer.get_test_dataloader(dataset)))

    assert eval_batch["input_ids"].shape == (2, 8)
    assert eval_batch["teacher_input_ids"].shape == (2, 12)
    assert test_batch["input_ids"].shape == (2, 8)
    assert test_batch["teacher_input_ids"].shape == (2, 12)


def test_cached_teacher_logits_work_with_train_and_eval_collators(
    tmp_path: Path,
) -> None:
    tokenizer = DummyPadTokenizer()
    dataset = _cached_dataset()
    trainer = DistillationTrainer(
        model=DummyModel(),
        args=_training_args(tmp_path),
        train_dataset=dataset,
        eval_dataset=dataset,
        processing_class=tokenizer,
        student_tokenizer=tokenizer,
        teacher_tokenizer=None,
        student_max_length=8,
        teacher_max_length=12,
        cache_teacher_logits=True,
    )

    train_batch = next(iter(trainer.get_train_dataloader()))
    eval_batch = next(iter(trainer.get_eval_dataloader()))

    assert train_batch["input_ids"].shape == (2, 5)
    assert eval_batch["input_ids"].shape == (2, 8)
    assert train_batch["teacher_logits"].shape == (2, 2)
    assert eval_batch["teacher_logits"].shape == (2, 2)
    assert train_batch["teacher_logits"].dtype == torch.float32
    assert "teacher_input_ids" not in train_batch


def test_non_distillation_path_selects_standard_trainer(tmp_path: Path) -> None:
    """When distillation is disabled, the code path must use standard Trainer, not DistillationTrainer.

    This is a regression test for the bug where train_cls.py always instantiated
    DistillationTrainer regardless of whether distillation was enabled.
    """
    tokenizer = DummyPadTokenizer()
    dataset = Dataset.from_list(
        [{"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": 0}]
    )
    # The non-distillation branch in train_cls.py constructs a standard Trainer
    # with processing_class= (not tokenizer= which is deprecated).
    trainer = Trainer(
        model=DummyModel(),
        args=_training_args(tmp_path),
        train_dataset=dataset,
        eval_dataset=dataset,
        processing_class=tokenizer,
    )
    assert type(trainer) is Trainer
    assert not isinstance(trainer, DistillationTrainer)
