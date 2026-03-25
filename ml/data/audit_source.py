#!/usr/bin/env python3
"""
PR-gateable source review helper for dataset pipeline.

Usage:
    python ml/data/audit_source.py --repo deepset/prompt-injections [--revision SHA] [--split train] [--limit 500]

Outputs (plain text to stdout, suitable for pasting into source audit notes):
- Total rows (after limit)
- Column names and dtypes
- Class balance after proposed label mapping
- Token length distribution: min, p25, p50, p75, p90, p99, max (whitespace tokenizer: len(text.split()))
- Token length distribution using bert-base-uncased tokenizer (if transformers available)
- Percentage of rows above 128 tokens
- Top 5 most common label values (raw, before mapping)
- Up to 5 random sample rows per detected label class
- Post-filter row estimate after label mapping and cap
- Class imbalance warning (>10:1) to stderr
- Source review checklist
"""

import argparse
import importlib.util
import random
import sys
from collections import Counter

_CHECKLIST = """=== Source Review Checklist ===
[ ] Schema documented (columns, dtypes above)
[ ] Label semantics reviewed manually (sample rows above)
[ ] License/terms approved for training use
[ ] Revision pinned in recipe
[ ] Loader test added to ml/data/tests/test_loaders.py
[ ] Split invariants still pass (run pytest ml/data/tests/)
[ ] Manifest contract still passes (run pytest ml/data/tests/)"""


def percentile(sorted_data, p):
    """Return the p-th percentile from sorted_data (0 <= p <= 100)."""
    n = len(sorted_data)
    if n == 0:
        return 0
    idx = (p / 100) * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return sorted_data[lo] * (1 - frac) + sorted_data[hi] * frac


def bert_token_lengths(texts, max_length=128):
    """Return sorted list of token lengths using bert-base-uncased tokenizer."""
    if importlib.util.find_spec("transformers") is None:
        print("[warn] transformers not installed — skipping token length distribution", file=sys.stderr)
        return None
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    lengths = []
    for text in texts:
        ids = tok.encode(str(text or ""), add_special_tokens=True, truncation=False)
        lengths.append(len(ids))
    return sorted(lengths), max_length


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="HuggingFace dataset repo (e.g. deepset/prompt-injections)")
    ap.add_argument("--revision", default=None, help="Git revision/SHA (optional)")
    ap.add_argument("--split", default="train", help="Dataset split to load (default: train)")
    ap.add_argument("--limit", type=int, default=500, help="Max rows to load (default: 500)")
    ap.add_argument("--cap", type=int, default=None, help="Post-filter cap for row estimate (optional)")
    args = ap.parse_args()

    print(f"=== SOURCE: {args.repo} ===")
    print(f"revision: {args.revision or 'latest'}")
    print(f"split: {args.split}")
    print(f"limit: {args.limit}")
    print()

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package not installed. Install with: pip install datasets", file=sys.stderr)
        sys.exit(1)

    try:
        ds = load_dataset(args.repo, revision=args.revision)
    except Exception as e:
        print(f"ERROR loading {args.repo}: {type(e).__name__}: {e}")
        print("\nAvailability: gated or unavailable")
        sys.exit(1)

    # Select split
    if args.split in ds:
        split = ds[args.split]
    elif "train" in ds:
        split = ds["train"]
        print(f"[note] Requested split '{args.split}' not found; using 'train'")
    else:
        split = list(ds.values())[0]
        print(f"[note] Requested split '{args.split}' not found; using first available split")

    split = split.flatten_indices().shuffle(seed=42)
    if args.limit and args.limit < len(split):
        split = split.select(range(args.limit))

    total = len(split)
    print(f"--- BASIC INFO ---")
    print(f"Total rows (after limit): {total}")
    print()

    # Schema: column names and dtypes
    print(f"--- SCHEMA ---")
    for col in split.column_names:
        feat = split.features[col]
        dtype = getattr(feat, "dtype", None) or type(feat).__name__
        print(f"  {col}: {dtype}")
    print()

    # Detect text column
    priority = ["text", "prompt", "input", "question"]
    text_col = None
    for k in priority:
        if k in split.column_names:
            text_col = k
            break
    if text_col is None:
        for col in split.column_names:
            if getattr(split.features[col], "dtype", None) == "string":
                text_col = col
                break
    if text_col is None and split.column_names:
        text_col = split.column_names[0]
    print(f"Detected text column: {text_col}")
    print()

    # Raw label values (before mapping)
    label_col = None
    for candidate in ["label", "labels", "class", "category", "type"]:
        if candidate in split.column_names:
            label_col = candidate
            break
    label_counts = Counter()
    print(f"--- RAW LABELS (before mapping) ---")
    if label_col:
        raw_labels = [str(x) for x in split[label_col]]
        label_counts = Counter(raw_labels)
        print(f"Label column: {label_col}")
        print(f"Top 5 most common raw label values:")
        for val, cnt in label_counts.most_common(5):
            print(f"  {val!r}: {cnt} ({100*cnt/total:.1f}%)")

        # Class imbalance warning
        counts_sorted = sorted(label_counts.values(), reverse=True)
        if len(counts_sorted) >= 2 and counts_sorted[-1] > 0:
            ratio = counts_sorted[0] / counts_sorted[-1]
            if ratio > 10:
                print(
                    f"[WARNING] Class imbalance ratio {ratio:.1f}:1 exceeds 10:1 threshold "
                    f"({counts_sorted[0]} vs {counts_sorted[-1]})",
                    file=sys.stderr,
                )
    else:
        print("No label column detected.")
    print()

    # Token length stats (whitespace)
    print(f"--- TOKEN LENGTH DISTRIBUTION (whitespace tokenizer) ---")
    if text_col:
        lengths = sorted([len(str(row.get(text_col, "") or "").split()) for row in split])
        pcts = [0, 25, 50, 75, 90, 99, 100]
        labels_pct = ["min", "p25", "p50", "p75", "p90", "p99", "max"]
        for label_p, p in zip(labels_pct, pcts):
            print(f"  {label_p}: {percentile(lengths, p):.1f}")
        above_128 = sum(1 for l in lengths if l > 128)
        print(f"  % rows > 128 tokens: {100*above_128/total:.1f}% ({above_128}/{total})")
    else:
        print("  Cannot compute — no text column detected.")
    print()

    # Token length stats (BERT tokenizer)
    print(f"--- TOKEN LENGTH DISTRIBUTION (bert-base-uncased, max_length=128) ---")
    if text_col:
        texts = [str(row.get(text_col, "") or "") for row in split]
        result = bert_token_lengths(texts)
        if result is not None:
            bert_lengths, max_length = result
            for label_p, p in zip(labels_pct, pcts):
                print(f"  {label_p}: {percentile(bert_lengths, p):.1f}")
            above_max = sum(1 for l in bert_lengths if l > max_length)
            print(f"  % rows > {max_length} tokens: {100*above_max/total:.1f}% ({above_max}/{total})")
        else:
            print("  Skipped (transformers unavailable).")
    else:
        print("  Cannot compute — no text column detected.")
    print()

    # Sample rows per label class (up to 5)
    print(f"--- SAMPLE ROWS (up to 5 per label class, seed=42) ---")
    if label_col and text_col:
        classes = sorted(label_counts.keys())
        random.seed(42)
        for cls in classes:
            class_rows = [row for row in split if str(row.get(label_col, "")) == cls]
            sample = random.sample(class_rows, min(5, len(class_rows)))
            print(f"\n  Label = {cls!r} ({label_counts[cls]} total):")
            for i, row in enumerate(sample, 1):
                text = str(row.get(text_col, ""))[:200]
                print(f"  [{i}] {text!r}")
    else:
        print("  Cannot sample — missing label or text column.")
    print()

    # Post-filter row estimate
    print(f"--- POST-FILTER ESTIMATE ---")
    if label_col and text_col:
        # Estimate rows that survive label mapping (non-empty label) and optional cap
        mapped_total = sum(label_counts.values())
        cap_applied = args.cap if args.cap else mapped_total
        estimated = min(mapped_total, cap_applied)
        print(f"  Rows with detected label: {mapped_total}")
        print(f"  Cap (--cap): {args.cap or 'none'}")
        print(f"  Estimated rows after mapping + cap: {estimated}")
    else:
        print("  Cannot estimate — missing label or text column.")
    print()

    print(f"=== END: {args.repo} ===")
    print()
    print(_CHECKLIST)


if __name__ == "__main__":
    main()
