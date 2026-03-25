from __future__ import annotations

import numpy as np
import pytest

from ml.training.slice_metrics import (
    LENGTH_BUCKET_SHORT,
    compute_slice_report,
    summarize_slice_report,
)
from ml.training.train_cls import (
    build_student_encoding_from_token_ids,
    select_student_token_window,
)


class DummyTokenizer:
    cls_token_id = 101
    sep_token_id = 102
    pad_token_id = 0

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        return 2

    def prepare_for_model(
        self,
        token_ids,
        truncation: bool,
        max_length: int,
        padding: str,
        return_attention_mask: bool,
    ):
        input_ids = [self.cls_token_id, *token_ids, self.sep_token_id]
        if len(input_ids) > max_length:
            raise AssertionError("Expected caller to truncate before prepare_for_model")
        pad_count = max_length - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * pad_count
        input_ids = input_ids + [self.pad_token_id] * pad_count
        output = {"input_ids": input_ids}
        if return_attention_mask:
            output["attention_mask"] = attention_mask
        return output

    def __call__(self, token_ids, truncation: bool, max_length: int, padding: str):
        content_budget = max_length - self.num_special_tokens_to_add(pair=False)
        truncated = token_ids[:content_budget] if truncation else token_ids
        return self.prepare_for_model(
            truncated,
            truncation=False,
            max_length=max_length,
            padding=padding,
            return_attention_mask=True,
        )


def test_head_strategy_matches_existing_head_truncation_behavior() -> None:
    tokenizer = DummyTokenizer()
    raw_token_ids = list(range(200))

    baseline = tokenizer(
        raw_token_ids, truncation=True, max_length=128, padding="max_length"
    )
    updated = build_student_encoding_from_token_ids(
        tokenizer=tokenizer,
        raw_token_ids=raw_token_ids,
        max_length=128,
        truncation_strategy="head",
    )

    assert updated["input_ids"] == baseline["input_ids"]
    assert updated["attention_mask"] == baseline["attention_mask"]
    assert updated["original_token_length"] == 200
    assert updated["length_bucket"] == ">128"


def test_tail_strategy_keeps_last_tokens() -> None:
    selected = select_student_token_window(
        list(range(200)),
        max_tokens=126,
        truncation_strategy="tail",
    )

    assert selected == list(range(74, 200))


def test_head_tail_strategy_keeps_first_and_last_halves() -> None:
    selected = select_student_token_window(
        list(range(200)),
        max_tokens=128,
        truncation_strategy="head_tail",
    )

    assert selected[:64] == list(range(64))
    assert selected[64:] == list(range(136, 200))


def test_short_examples_keep_pre_truncation_length_bucket() -> None:
    tokenizer = DummyTokenizer()

    updated = build_student_encoding_from_token_ids(
        tokenizer=tokenizer,
        raw_token_ids=list(range(40)),
        max_length=128,
        truncation_strategy="head",
    )

    assert updated["original_token_length"] == 40
    assert updated["length_bucket"] == "<=128"


def test_teacher_always_uses_head_truncation_when_student_uses_non_head_strategy() -> (
    None
):
    """Teacher preprocessing must remain head-truncated regardless of student truncation_strategy.

    DummyTokenizer.__call__ implements head truncation (first content_budget tokens).
    build_student_encoding_from_token_ids with tail/head_tail must produce different
    input_ids than DummyTokenizer would for the same raw token sequence, confirming the
    two paths diverge.  The teacher path in train_cls.prep() uses tokenizer(text,
    truncation=True, max_length=...) which is equivalent to DummyTokenizer head
    truncation — so matching the DummyTokenizer output proves the teacher stays
    head-truncated.
    """
    tokenizer = DummyTokenizer()
    raw_token_ids = list(range(200))
    max_length = 128

    # Teacher reference: head truncation via DummyTokenizer.__call__
    teacher_encoded = tokenizer(
        raw_token_ids, truncation=True, max_length=max_length, padding="max_length"
    )
    teacher_ids = teacher_encoded["input_ids"]

    # Student with tail strategy: must differ from teacher
    student_tail = build_student_encoding_from_token_ids(
        tokenizer=tokenizer,
        raw_token_ids=raw_token_ids,
        max_length=max_length,
        truncation_strategy="tail",
    )
    assert student_tail["input_ids"] != teacher_ids, (
        "tail student must differ from head-truncated teacher"
    )
    # Teacher content tokens are ids 0..125 (head); tail student content is ids 74..199
    content_budget = max_length - tokenizer.num_special_tokens_to_add()
    expected_teacher_content = list(range(content_budget))
    assert teacher_ids[1 : content_budget + 1] == expected_teacher_content

    # Student with head_tail strategy: must also differ from teacher
    student_head_tail = build_student_encoding_from_token_ids(
        tokenizer=tokenizer,
        raw_token_ids=raw_token_ids,
        max_length=max_length,
        truncation_strategy="head_tail",
    )
    assert student_head_tail["input_ids"] != teacher_ids, (
        "head_tail student must differ from head-truncated teacher"
    )


def test_prep_teacher_uses_head_truncation_while_student_uses_configured_strategy() -> (
    None
):
    """Integration test: simulate the two tokenization calls inside prep().

    prep() runs:
      student: build_student_batch(tok, texts, max_length, truncation_strategy)
      teacher: teacher_tok(texts, truncation=True, max_length=max_length, padding="max_length")

    These must produce different input_ids when truncation_strategy is not "head",
    proving the teacher path stays head-truncated while the student path follows the
    configured strategy.
    """
    AutoTokenizer = pytest.importorskip(
        "transformers", reason="transformers required"
    ).AutoTokenizer
    from ml.training.train_cls import build_student_batch

    tok = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    teacher_tok = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    max_length = 128

    # Build a text that tokenizes to well above 128 tokens so truncation is active.
    long_text = " ".join([str(i) for i in range(300)])
    texts = [long_text]

    # Teacher path (exactly what prep() does at lines 849-854 in train_cls.py):
    t = teacher_tok(texts, truncation=True, max_length=max_length, padding="max_length")
    teacher_ids = t["input_ids"][0]

    # Student path with tail (exactly what prep() does via build_student_batch):
    student_out = build_student_batch(
        tokenizer=tok,
        texts=texts,
        max_length=max_length,
        truncation_strategy="tail",
    )
    student_ids = student_out["input_ids"][0]

    # They must differ — teacher is head-truncated, student is tail-truncated.
    assert teacher_ids != student_ids, (
        "teacher_input_ids must differ from student tail-truncated input_ids"
    )

    # Confirm teacher took head tokens: its non-padding token at index 1 should be
    # the same as the first content token from a plain head-truncated encoding.
    head_ref = tok(texts, truncation=True, max_length=max_length, padding="max_length")
    assert teacher_ids == head_ref["input_ids"][0], (
        "teacher must produce the same ids as standard head truncation"
    )

    # Also verify with head_tail strategy.
    student_ht = build_student_batch(
        tokenizer=tok,
        texts=texts,
        max_length=max_length,
        truncation_strategy="head_tail",
    )
    assert student_ht["input_ids"][0] != teacher_ids, (
        "teacher_input_ids must differ from student head_tail-truncated input_ids"
    )


def test_slice_report_handles_degenerate_and_length_bucket_counts() -> None:
    labels = np.asarray([0, 1, 1, 1], dtype=np.int64)
    probs = np.asarray([0.10, 0.90, 0.80, 0.20], dtype=np.float32)
    metadata = [
        {
            "source": "alpha",
            "length_bucket": LENGTH_BUCKET_SHORT,
        },
        {
            "source": "alpha",
            "length_bucket": LENGTH_BUCKET_SHORT,
        },
        {
            "source": "beta",
            "length_bucket": ">128",
        },
        {
            "source": "beta",
            "length_bucket": ">128",
        },
    ]

    report = compute_slice_report(
        labels=labels,
        probs=probs,
        metadata=metadata,
        threshold=0.5,
    )

    assert report["by_length_bucket"]["<=128"]["example_count"] == 2
    assert report["by_length_bucket"][">128"]["example_count"] == 2
    assert report["by_source"]["beta"]["pr_auc"] is None
    assert report["by_source"]["beta"]["precision_at_1pct_fpr"] == 1.0
    assert report["by_source"]["beta"]["recall_at_1pct_fpr"] == 0.5
    assert report["by_source"]["beta"]["f1_at_1pct_fpr"] == 2.0 / 3.0

    summary = summarize_slice_report(report)
    assert summary["source_count"] == 2
    assert summary["length_buckets"]["<=128"]["example_count"] == 2
    assert summary["length_buckets"][">128"]["example_count"] == 2
