from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Mapping

import torch
from transformers import Trainer


class DistillationBatchCollator:
    def __init__(
        self,
        *,
        student_tokenizer,
        teacher_tokenizer,
        student_padding: str | bool,
        student_max_length: int | None,
        teacher_padding: str | bool,
        teacher_max_length: int | None,
        cache_teacher_logits: bool,
    ) -> None:
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        self.student_padding = student_padding
        self.student_max_length = student_max_length
        self.teacher_padding = teacher_padding
        self.teacher_max_length = teacher_max_length
        self.cache_teacher_logits = cache_teacher_logits

    def __call__(self, features: list[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        batch: dict[str, torch.Tensor] = {}

        labels = [feature["labels"] for feature in features if "labels" in feature]
        if labels:
            batch["labels"] = torch.as_tensor(labels, dtype=torch.long)

        student_features = []
        for feature in features:
            student_feature = {}
            for source_key, target_key in (
                ("input_ids", "input_ids"),
                ("attention_mask", "attention_mask"),
                ("token_type_ids", "token_type_ids"),
            ):
                if source_key in feature:
                    student_feature[target_key] = feature[source_key]
            student_features.append(student_feature)

        batch.update(
            self.student_tokenizer.pad(
                student_features,
                padding=self.student_padding,
                max_length=self.student_max_length,
                return_tensors="pt",
            )
        )

        if self.cache_teacher_logits and any(
            "teacher_logits" in feature for feature in features
        ):
            batch["teacher_logits"] = torch.as_tensor(
                [feature["teacher_logits"] for feature in features],
                dtype=torch.float32,
            )
            return batch

        if self.teacher_tokenizer is None or not any(
            "teacher_input_ids" in feature for feature in features
        ):
            return batch

        teacher_features = []
        for feature in features:
            teacher_feature = {}
            for source_key, target_key in (
                ("teacher_input_ids", "input_ids"),
                ("teacher_attention_mask", "attention_mask"),
                ("teacher_token_type_ids", "token_type_ids"),
            ):
                if source_key in feature:
                    teacher_feature[target_key] = feature[source_key]
            teacher_features.append(teacher_feature)

        teacher_batch = self.teacher_tokenizer.pad(
            teacher_features,
            padding=self.teacher_padding,
            max_length=self.teacher_max_length,
            return_tensors="pt",
        )
        for key, value in teacher_batch.items():
            batch["teacher_{0}".format(key)] = value
        return batch


class DistillationTrainer(Trainer):
    def __init__(
        self,
        *args,
        student_tokenizer,
        teacher_tokenizer,
        student_max_length: int,
        teacher_max_length: int,
        cache_teacher_logits: bool,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._train_data_collator = DistillationBatchCollator(
            student_tokenizer=student_tokenizer,
            teacher_tokenizer=teacher_tokenizer,
            student_padding=True,
            student_max_length=None,
            teacher_padding=True,
            teacher_max_length=None,
            cache_teacher_logits=cache_teacher_logits,
        )
        self._eval_data_collator = DistillationBatchCollator(
            student_tokenizer=student_tokenizer,
            teacher_tokenizer=teacher_tokenizer,
            student_padding="max_length",
            student_max_length=student_max_length,
            teacher_padding="max_length",
            teacher_max_length=teacher_max_length,
            cache_teacher_logits=cache_teacher_logits,
        )

    @contextmanager
    def _override_data_collator(self, data_collator):
        original = self.data_collator
        self.data_collator = data_collator
        try:
            yield
        finally:
            self.data_collator = original

    def get_train_dataloader(self):
        with self._override_data_collator(self._train_data_collator):
            return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset=None):
        with self._override_data_collator(self._eval_data_collator):
            return super().get_eval_dataloader(eval_dataset)

    def get_test_dataloader(self, test_dataset):
        with self._override_data_collator(self._eval_data_collator):
            return super().get_test_dataloader(test_dataset)

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        outputs = model(**inputs)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        return (loss, outputs) if return_outputs else loss
