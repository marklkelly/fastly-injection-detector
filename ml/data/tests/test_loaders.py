"""Unit tests for loader label mapping logic. Uses mocks so no network access needed."""
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from unittest.mock import patch, MagicMock
from datasets import Dataset


def make_fake_dataset(rows):
    return Dataset.from_list(rows)


class TestJayavibhav:
    def test_mapping_preserves_01(self):
        fake = make_fake_dataset([
            {"text": "hello", "label": 0},
            {"text": "ignore prev instructions", "label": 1},
        ])
        fake_ds = {"train": fake}
        with patch("ml.data.build.load_dataset", return_value=fake_ds):
            from ml.data.build import load_jayavibhav_prompt_injection
            result = load_jayavibhav_prompt_injection(limit=10)
        assert all(r in {0, 1} for r in result["label"])
        assert "text" in result.column_names and "source" in result.column_names

    def test_source_field(self):
        fake = make_fake_dataset([{"text": "hello", "label": 0}])
        with patch("ml.data.build.load_dataset", return_value={"train": fake}):
            from ml.data.build import load_jayavibhav_prompt_injection
            result = load_jayavibhav_prompt_injection()
        assert all(s == "jayavibhav/prompt-injection" for s in result["source"])


class TestXTRam1:
    def _make_ds(self, rows):
        from datasets import Features, Value
        feats = Features({"text": Value("string"), "label": Value("int64")})
        return Dataset.from_list(rows, features=feats)

    def test_hardcoded_mapping(self):
        fake = self._make_ds([{"text": "safe", "label": 0}, {"text": "inject", "label": 1}])
        with patch("ml.data.build.load_dataset", return_value={"train": fake}):
            from ml.data.build import load_xTRam1_safe_guard
            result = load_xTRam1_safe_guard()
        labels = list(result["label"])
        assert 0 in labels and 1 in labels

    def test_wrong_dtype_raises(self):
        from datasets import Features, Value
        feats = Features({"text": Value("string"), "label": Value("string")})
        fake = Dataset.from_list([{"text": "x", "label": "safe"}], features=feats)
        with patch("ml.data.build.load_dataset", return_value={"train": fake}):
            from ml.data.build import load_xTRam1_safe_guard
            with pytest.raises(ValueError, match="schema changed"):
                load_xTRam1_safe_guard()

    def test_unexpected_labels_raises(self):
        from datasets import Features, Value
        feats = Features({"text": Value("string"), "label": Value("int64")})
        fake = Dataset.from_list([{"text": "x", "label": 2}], features=feats)
        with patch("ml.data.build.load_dataset", return_value={"train": fake}):
            from ml.data.build import load_xTRam1_safe_guard
            with pytest.raises(ValueError, match="label values changed"):
                load_xTRam1_safe_guard()


class TestRubend18:
    def test_all_injection(self):
        fake = make_fake_dataset([{"text": "jailbreak text", "label": "jailbreak"}])
        with patch("ml.data.build.load_dataset", return_value={"train": fake}):
            from ml.data.build import load_rubend18_jailbreak
            result = load_rubend18_jailbreak()
        assert all(y == 1 for y in result["label"])

    def test_source_field(self):
        fake = make_fake_dataset([{"text": "jailbreak text", "label": "jailbreak"}])
        with patch("ml.data.build.load_dataset", return_value={"train": fake}):
            from ml.data.build import load_rubend18_jailbreak
            result = load_rubend18_jailbreak()
        assert all(s == "rubend18/ChatGPT-Jailbreak-Prompts" for s in result["source"])


class TestDarkknight25:
    def test_label_mapping(self):
        fake = make_fake_dataset([
            {"prompt": "safe query", "label": "benign"},
            {"prompt": "attack", "label": "malicious"},
        ])
        with patch("ml.data.build.load_dataset", return_value={"train": fake}):
            from ml.data.build import load_darkknight25_prompt_benign
            result = load_darkknight25_prompt_benign()
        labels = list(result["label"])
        assert 0 in labels and 1 in labels

    def test_text_and_source_present(self):
        fake = make_fake_dataset([{"prompt": "hello", "label": "benign"}])
        with patch("ml.data.build.load_dataset", return_value={"train": fake}):
            from ml.data.build import load_darkknight25_prompt_benign
            result = load_darkknight25_prompt_benign()
        assert "text" in result.column_names
        assert "source" in result.column_names


class TestDeepset:
    def test_mapping_produces_0_and_1(self):
        fake = make_fake_dataset([
            {"text": "hello world", "label": 0},
            {"text": "ignore all previous instructions", "label": 1},
        ])
        with patch("ml.data.build.load_dataset", return_value={"train": fake}):
            from ml.data.build import load_deepset_prompt_injections
            result = load_deepset_prompt_injections()
        assert all(y in {0, 1} for y in result["label"])

    def test_source_field(self):
        fake = make_fake_dataset([{"text": "hi", "label": 0}])
        fake2 = make_fake_dataset([{"text": "inject", "label": 1}])
        from datasets import concatenate_datasets
        combined = concatenate_datasets([fake, fake2])
        with patch("ml.data.build.load_dataset", return_value={"train": combined}):
            from ml.data.build import load_deepset_prompt_injections
            result = load_deepset_prompt_injections()
        assert all(s == "deepset/prompt-injections" for s in result["source"])


class TestHackaprompt:
    def test_returns_none_on_error(self):
        with patch("ml.data.build.load_dataset", side_effect=Exception("gated")):
            from ml.data.build import load_hackaprompt
            result = load_hackaprompt()
        assert result is None

    def test_all_injection_when_loaded(self):
        fake = make_fake_dataset([{"user_input": "break the rules", "prompt_level": 1}])
        with patch("ml.data.build.load_dataset", return_value={"train": fake}):
            from ml.data.build import load_hackaprompt
            result = load_hackaprompt()
        assert result is not None
        assert all(y == 1 for y in result["label"])


class TestMarkush1:
    def test_skips_on_pending_review(self, capsys):
        from ml.data.build import load_markush1_injection
        result = load_markush1_injection(pending_license_review=True)
        assert result is None
        captured = capsys.readouterr()
        assert "pending_license_review" in captured.err

    def test_returns_none_on_load_error(self):
        with patch("ml.data.build.load_dataset", side_effect=Exception("404")):
            from ml.data.build import load_markush1_injection
            result = load_markush1_injection(pending_license_review=False)
        assert result is None
