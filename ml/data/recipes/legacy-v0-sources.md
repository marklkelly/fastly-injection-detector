# Legacy v0 Dataset Sources (2025-09-18)

This documents the sources used in the original `ml/training/data_out/` build
before the recipe-driven pipeline was introduced. Preserved before cleanup.

## Build metadata
- Build ID: 2025-09-18T11:50:59Z
- Tokenizer: bert-base-uncased
- Max tokens: 128
- Total rows (union_raw): 629,777
- After dedup (union_clean_dedup): 389,539
- Final training set (train_balanced_len128_filtered): 66,041

## Sources

| Source | Rows in clean-dedup | Notes |
|--------|---------------------|-------|
| hendzh/PromptShield | 18,782 | Used in v0; NOT in pi_mix_v1 or pi_mix_v2 — dropped due to supersession by jayavibhav/prompt-injection |
| xTRam1/safe-guard-prompt-injection | 1 | Near-zero contribution; replaced by hardcoded loader in v1/v2 |
| deepset/prompt-injections | 546 | Now used as OOD-only source in pi_mix_v2 |
| darkknight25/Prompt_Injection_Benign_Prompt_Dataset | 249 | Retained in pi_mix_v1 and v2 |
| hackaprompt/hackaprompt-dataset | 369,891 | Retained in pi_mix_v2 (capped, in-domain) |
| rubend18/ChatGPT-Jailbreak-Prompts | 70 | Retained as Harelix fallback in v1/v2 |

## Key differences from pi_mix_v2
- `hendzh/PromptShield` was the largest source at 18k rows; it was replaced by
  `jayavibhav/prompt-injection` (capped at 30k) in pi_mix_v1/v2
- No `allenai/wildjailbreak` or `markush1/LLM-Injection-Dataset` in v0
- No dedup beyond basic exact dedup; no MinHash clustering
- No cluster-aware splitting; no `test_ood.jsonl`
- Thresholds were calibrated on a different validation distribution
