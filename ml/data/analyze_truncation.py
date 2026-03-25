#!/usr/bin/env python3
"""
Analyze token length distribution in a dataset to understand truncation risk
at the 128-token deployment boundary.

Usage:
    python ml/data/analyze_truncation.py --data ml/data/versions/pi_mix_v1/train.jsonl [--max-len 128]
"""

import argparse
import json
import random
import sys
from collections import Counter


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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="Path to .jsonl file")
    ap.add_argument("--max-len", type=int, default=128, help="Truncation boundary (default: 128)")
    ap.add_argument(
        "--tokenizer",
        default="whitespace",
        help="'whitespace' (default) or a HuggingFace tokenizer name. "
             "Currently only whitespace proxy is implemented.",
    )
    args = ap.parse_args()

    rows = []
    try:
        with open(args.data, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    except FileNotFoundError:
        print(f"ERROR: File not found: {args.data}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSONL at {args.data}: {e}", file=sys.stderr)
        sys.exit(1)

    if not rows:
        print("ERROR: No rows loaded.", file=sys.stderr)
        sys.exit(1)

    print(f"=== TRUNCATION ANALYSIS ===")
    print(f"File: {args.data}")
    print(f"Tokenizer: {args.tokenizer} (whitespace proxy: len(text.split()))")
    print(f"Max length boundary: {args.max_len}")
    print(f"Total rows: {len(rows)}")
    print()

    # Compute token lengths
    lengths = []
    inj_lengths = []
    safe_lengths = []
    for r in rows:
        text = str(r.get("text", ""))
        n = len(text.split())
        label = r.get("label", -1)
        lengths.append((n, label))
        if label == 1:
            inj_lengths.append(n)
        elif label == 0:
            safe_lengths.append(n)

    all_lens = sorted(l for l, _ in lengths)

    print("--- TOKEN LENGTH PERCENTILES ---")
    for label_p, p in [("p10", 10), ("p25", 25), ("p50", 50), ("p75", 75), ("p90", 90), ("p95", 95), ("p99", 99), ("max", 100)]:
        print(f"  {label_p}: {percentile(all_lens, p):.0f}")
    print()

    # Histogram in 10 buckets
    print("--- HISTOGRAM (bucket size = 20 tokens) ---")
    buckets = [0] * 10
    overflow = 0
    for l in all_lens:
        bucket = l // 20
        if bucket < 10:
            buckets[bucket] += 1
        else:
            overflow += 1
    for i, cnt in enumerate(buckets):
        lo = i * 20
        hi = lo + 19
        bar = "#" * min(40, cnt * 40 // max(1, max(buckets)))
        print(f"  {lo:3d}-{hi:3d}: {cnt:6d}  {bar}")
    if overflow:
        print(f"  200+  : {overflow:6d}")
    print()

    # Truncation stats
    total = len(rows)
    n_above = sum(1 for l in all_lens if l > args.max_len)
    n_inj_above = sum(1 for l in inj_lengths if l > args.max_len)
    n_safe_above = sum(1 for l in safe_lengths if l > args.max_len)

    print(f"--- TRUNCATION RISK (> {args.max_len} tokens) ---")
    print(f"  % total examples > {args.max_len}: {100*n_above/total:.1f}% ({n_above}/{total})")
    if inj_lengths:
        print(f"  % INJECTION (label=1) > {args.max_len}: {100*n_inj_above/len(inj_lengths):.1f}% ({n_inj_above}/{len(inj_lengths)})")
    if safe_lengths:
        print(f"  % SAFE (label=0) > {args.max_len}: {100*n_safe_above/len(safe_lengths):.1f}% ({n_safe_above}/{len(safe_lengths)})")
    print()

    # Sample long INJECTION examples
    print(f"--- 5 RANDOM INJECTION EXAMPLES > {args.max_len} TOKENS ---")
    long_inj = [r for r in rows if r.get("label") == 1 and len(str(r.get("text", "")).split()) > args.max_len]
    random.seed(42)
    sample = random.sample(long_inj, min(5, len(long_inj)))
    if sample:
        for i, r in enumerate(sample, 1):
            text = str(r.get("text", ""))[:200]
            n = len(r["text"].split())
            print(f"  [{i}] ({n} tokens) {text!r}")
    else:
        print("  None found.")
    print()

    # Policy recommendation
    if inj_lengths:
        pct_inj_above = 100 * n_inj_above / len(inj_lengths)
        print("--- POLICY RECOMMENDATION ---")
        if pct_inj_above > 15:
            print(
                f"  WARNING: {pct_inj_above:.1f}% of INJECTION examples exceed {args.max_len} tokens. "
                "Consider tightening max_chars in the recipe to reduce truncation risk."
            )
        else:
            print(
                f"  OK: {pct_inj_above:.1f}% of INJECTION examples exceed {args.max_len} tokens. "
                "Truncation risk is within acceptable range."
            )
    print()
    print("=== END TRUNCATION ANALYSIS ===")


if __name__ == "__main__":
    main()
