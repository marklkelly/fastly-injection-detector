# Manual Edge-Case Corpus

These are scaffolding examples for `edge_case_corpus_v1.jsonl`. 
**The developer should expand each category before training.**

## Schema
{"text": "...", "label": 0|1, "source": "local/edge_case_corpus_v1", "category": "..."}

Labels: 0=SAFE, 1=INJECTION

## Categories
- obfuscation: character substitution, leetspeak, spacing tricks (label=1)
- multilingual: non-English injections or mixed scripts (label=1)
- indirect_document: injections embedded in documents (label=1)
- tool_use: targeting agent/function-call context (label=1)
- unicode_steganography: homoglyphs, zero-width chars (label=1)
- safe_near_miss: benign prompts resembling injections (label=0)

## Status
Scaffold only. Each category has ≥10 placeholder examples.
Expand to 100–200 examples per category before training v2.
