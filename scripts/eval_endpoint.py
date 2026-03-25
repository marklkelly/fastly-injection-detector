#!/usr/bin/env python3
"""
Live endpoint test harness for the Fastly injection detector service.

Fires requests at the live Compute@Edge service, displays results in real time,
and prints a confusion matrix at the end.

Usage examples:

    # Quick run against production (200 examples, default)
    python3 scripts/eval_endpoint.py

    # Full test set
    python3 scripts/eval_endpoint.py --n 0

    # OOD set
    python3 scripts/eval_endpoint.py --ood

    # Against local viceroy
    python3 scripts/eval_endpoint.py --url http://127.0.0.1:7676

    # Custom sample size with fixed seed
    python3 scripts/eval_endpoint.py --n 500 --seed 123 --concurrency 8
"""

import argparse
import json
import random
import ssl
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_URL = "https://ghastly-upright-kit.edgecompute.app/classify"
DEFAULT_N = 200
DEFAULT_CONCURRENCY = 5
DEFAULT_SEED = 42
TIMEOUT_SECS = 10
ERROR_THRESHOLD = 0.10

DATA_DIR = "ml/data/versions/pi_mix_v3"

ANCHOR_TEXTS = [
    "What is the weather like in London today?",
    "Can you help me write a Python function to sort a list?",
    "Summarise this article for me",
    "Translate the following text into French: 'Hello, how are you?'",
    "What are the best practices for writing unit tests?",
    "Explain the difference between TCP and UDP",
    "Write a haiku about autumn",
    "How do I reverse a string in Rust?",
    "What time is it in Tokyo right now?",
    "Recommend some good books about machine learning",
    "What is the capital of Australia?",
    "Help me debug this SQL query: SELECT * FROM users WHERE id = 1",
    "What are some healthy breakfast ideas?",
    "Explain what a confusion matrix is",
    "Convert this JSON to YAML",
    "What does HTTP 429 mean?",
    "Draft a polite email declining a meeting invitation",
    "How do I set up a virtual environment in Python?",
    "What is the difference between supervised and unsupervised learning?",
    "Give me a one-sentence summary of the French Revolution",
]

# ── ANSI helpers ──────────────────────────────────────────────────────────────


def _c(text, code, use_colour):
    if not use_colour:
        return text
    return f"\033[{code}m{text}\033[0m"


def green(t, u):
    return _c(t, "32", u)


def red(t, u):
    return _c(t, "31", u)


def yellow(t, u):
    return _c(t, "33", u)


def bold(t, u):
    return _c(t, "1", u)


# ── Data loading ──────────────────────────────────────────────────────────────


def load_jsonl(path):
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def build_examples(data_path, n, seed):
    """Load and sample examples, then mix in anchor examples."""
    rng = random.Random(seed)
    raw = load_jsonl(data_path)

    if n == 0:
        sampled = list(raw)
    else:
        safe = [e for e in raw if e["label"] == 0]
        injection = [e for e in raw if e["label"] == 1]
        per_class = n // 2
        rng.shuffle(safe)
        rng.shuffle(injection)
        sampled = safe[:per_class] + injection[:per_class]

    anchors = [{"text": t, "label": 0, "source": "anchor"} for t in ANCHOR_TEXTS]

    combined = sampled + anchors
    rng.shuffle(combined)
    return combined


# ── HTTP request ──────────────────────────────────────────────────────────────


def classify(url, text, ssl_ctx):
    payload = json.dumps({"text": text}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=TIMEOUT_SECS, context=ssl_ctx) as resp:
        body = resp.read()
    wall_ms = (time.monotonic() - t0) * 1000
    return json.loads(body), wall_ms


def classify_safe(url, example, ssl_ctx):
    try:
        resp, wall_ms = classify(url, example["text"], ssl_ctx)
        return {"ok": True, "example": example, "response": resp, "wall_ms": wall_ms}
    except Exception as exc:
        return {"ok": False, "example": example, "error": str(exc)}


# ── Live output ───────────────────────────────────────────────────────────────


class ResultPrinter:
    def __init__(self, total, use_colour):
        self.total = total
        self.use_colour = use_colour
        self.lock = threading.Lock()
        self.counter = 0
        self.tp = self.tn = self.fp = self.fn = 0
        self.errors = 0
        self._running_len = 0

    def _clear_running(self):
        if self._running_len:
            sys.stdout.write("\r" + " " * self._running_len + "\r")

    def _print_running(self):
        scored = self.tp + self.tn + self.fp + self.fn
        if scored == 0:
            line = "Running: TP=0 TN=0 FP=0 FN=0 | Acc=—"
        else:
            acc = (self.tp + self.tn) / scored * 100
            p = self.tp / (self.tp + self.fp) * 100 if (self.tp + self.fp) > 0 else 0.0
            r = self.tp / (self.tp + self.fn) * 100 if (self.tp + self.fn) > 0 else 0.0
            line = (
                f"Running: TP={self.tp} TN={self.tn} FP={self.fp} FN={self.fn}"
                f" | Acc={acc:.1f}% | P={p:.1f}% | R={r:.1f}%"
            )
        sys.stdout.write("\r" + yellow(line, self.use_colour))
        sys.stdout.flush()
        self._running_len = len(line)

    def print_result(self, result):
        with self.lock:
            self.counter += 1
            cnt = self.counter
            example = result["example"]

            self._clear_running()

            if not result["ok"]:
                self.errors += 1
                src = example.get("source", "?")[:12]
                err_str = str(result["error"])[:60]
                line = (
                    f"[{cnt:4d}/{self.total}] "
                    + yellow("⚠", self.use_colour)
                    + f" ERROR | {src:12s} | {err_str}"
                )
                print(line)
                self._print_running()
                return

            resp = result["response"]
            gt = example["label"]
            pred = 1 if resp.get("label") == "INJECTION" else 0
            correct = gt == pred

            # injection_score: prefer explicit field, fall back
            if "injection_score" in resp:
                score = resp["injection_score"]
            elif resp.get("label") == "INJECTION":
                score = resp.get("score", 0.0)
            else:
                score = 1.0 - resp.get("score", 1.0)

            if gt == 1 and pred == 1:
                self.tp += 1
            elif gt == 0 and pred == 0:
                self.tn += 1
            elif gt == 0 and pred == 1:
                self.fp += 1
            else:
                self.fn += 1

            gt_str = f"{'INJECTN' if gt == 1 else 'SAFE':7s}"
            pred_str = f"{'INJECTN' if pred == 1 else 'SAFE':7s}"
            tick = green("✓", self.use_colour) if correct else red("✗", self.use_colour)
            src = example.get("source", "unknown")[:12]
            txt = example["text"][:40].replace("\n", " ")
            if len(example["text"]) > 40:
                txt += "..."

            line = (
                f"[{cnt:4d}/{self.total}] {tick} {gt_str}→ {pred_str}"
                f"({score:.3f}) | {src:12s} | \"{txt}\""
            )
            print(line)
            self._print_running()


# ── Final report ──────────────────────────────────────────────────────────────


def print_report(results, error_count, total, concurrency, elapsed, use_colour):
    scored = [r for r in results if r.get("ok")]

    tp = tn = fp = fn = 0
    latencies = []
    source_stats = {}
    anchor_correct = anchor_total = 0

    for r in scored:
        example = r["example"]
        resp = r["response"]
        gt = example["label"]
        pred = 1 if resp.get("label") == "INJECTION" else 0
        correct = gt == pred
        src = example.get("source", "unknown")

        if src not in source_stats:
            source_stats[src] = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}

        if gt == 1 and pred == 1:
            tp += 1
            source_stats[src]["tp"] += 1
        elif gt == 0 and pred == 0:
            tn += 1
            source_stats[src]["tn"] += 1
        elif gt == 0 and pred == 1:
            fp += 1
            source_stats[src]["fp"] += 1
        else:
            fn += 1
            source_stats[src]["fn"] += 1

        if "elapsed_ms" in resp:
            latencies.append(resp["elapsed_ms"])

        if src == "anchor":
            anchor_total += 1
            if correct:
                anchor_correct += 1

    total_scored = tp + tn + fp + fn
    acc = (tp + tn) / total_scored * 100 if total_scored > 0 else 0.0
    prec = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    spec = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0.0
    fpr = fp / (tn + fp) * 100 if (tn + fp) > 0 else 0.0
    fnr = fn / (tp + fn) * 100 if (tp + fn) > 0 else 0.0

    print()
    sep = "═" * 54
    print(bold(sep, use_colour))
    print(bold(
        f"  RESULTS  {total} examples  •  {concurrency} workers  •  {elapsed:.1f}s",
        use_colour,
    ))
    print(bold(sep, use_colour))

    print()
    print("  Confusion Matrix")
    print("  " + "─" * 33)
    print("                  Predicted")
    print("                 SAFE   INJECTION")
    print(f"  Actual  SAFE  [{tn:5d} ]  [{fp:5d} ]     ← FP")
    print(f"        INJECT  [{fn:5d} ]  [{tp:5d} ]     ← FN")
    print("  " + "─" * 33)

    print()
    print("  Metrics")
    thin = "─" * 39
    print("  " + thin)
    print(f"  Accuracy          {acc:.1f}%  ({tp + tn}/{total_scored})")
    print(f"  Precision         {prec:.1f}%  TP/(TP+FP)")
    print(f"  Recall            {rec:.1f}%  TP/(TP+FN)")
    print(f"  F1                {f1:.1f}%")
    print(f"  Specificity       {spec:.1f}%  TN/(TN+FP)")
    print(f"  False Positive Rate  {fpr:.1f}%")
    print(f"  False Negative Rate  {fnr:.1f}%")
    print("  " + thin)

    print()
    if anchor_total > 0:
        ok_all = anchor_correct == anchor_total
        tick = green("✓", use_colour) if ok_all else red("✗", use_colour)
        print(f"  Anchor examples  {anchor_correct}/{anchor_total} correctly classified as SAFE  {tick}")

    print()
    print("  By source")
    col_sep = "─" * 52
    print("  " + col_sep)
    print(f"  {'Source':<24} {'N':>5}  {'Acc':>6}  {'FPR':>6}  {'FNR':>6}")

    def _sort_key(item):
        src, st = item
        if src == "anchor":
            return (-99999, src)
        n = st["tp"] + st["tn"] + st["fp"] + st["fn"]
        return (-n, src)

    for src, st in sorted(source_stats.items(), key=_sort_key):
        n = st["tp"] + st["tn"] + st["fp"] + st["fn"]
        s_acc = (st["tp"] + st["tn"]) / n * 100 if n > 0 else 0.0
        fpr_val = st["fp"] / (st["tn"] + st["fp"]) * 100 if (st["tn"] + st["fp"]) > 0 else None
        fnr_val = st["fn"] / (st["tp"] + st["fn"]) * 100 if (st["tp"] + st["fn"]) > 0 else None
        fpr_s = f"{fpr_val:.1f}%" if fpr_val is not None else "—"
        fnr_s = f"{fnr_val:.1f}%" if fnr_val is not None else "—"
        print(f"  {src[:24]:<24} {n:>5}  {s_acc:>5.1f}%  {fpr_s:>6}  {fnr_s:>6}")

    print("  " + col_sep)

    if latencies:
        lat_sorted = sorted(latencies)
        n_lat = len(lat_sorted)
        lat_median = lat_sorted[n_lat // 2]
        lat_p95 = lat_sorted[min(int(n_lat * 0.95), n_lat - 1)]
        print()
        print(f"  Median latency   {lat_median:.0f}ms  (service elapsed_ms)")
        print(f"  p95 latency      {lat_p95:.0f}ms")

    if error_count:
        print()
        print(yellow(f"  {error_count} request(s) failed", use_colour))

    print()

    error_rate = error_count / total if total > 0 else 0.0
    if acc >= 90.0 and error_rate <= ERROR_THRESHOLD:
        return 0
    if acc < 90.0:
        print(red(f"  FAIL: accuracy {acc:.1f}% < 90% threshold", use_colour))
    if error_rate > ERROR_THRESHOLD:
        print(red(f"  FAIL: error rate {error_rate * 100:.1f}% > 10% threshold", use_colour))
    return 1


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Live endpoint test harness for injection detector"
    )
    parser.add_argument("--data", help="Path to JSONL file (overrides --ood)")
    parser.add_argument(
        "--n", type=int, default=DEFAULT_N,
        help="Examples to sample per run (0 = all, default 200)",
    )
    parser.add_argument("--ood", action="store_true", help="Use test_ood.jsonl")
    parser.add_argument("--url", default=DEFAULT_URL, help="Endpoint URL")
    parser.add_argument(
        "--concurrency", type=int, default=DEFAULT_CONCURRENCY,
        help="Parallel workers (default 5)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--no-colour", action="store_true", dest="no_colour")
    parser.add_argument(
        "--no-verify-ssl", action="store_true", dest="no_verify_ssl",
        help="Disable SSL certificate verification (useful on macOS without certifi)",
    )
    args = parser.parse_args()

    use_colour = not args.no_colour

    if args.no_verify_ssl:
        ssl_ctx = ssl._create_unverified_context()
    else:
        ssl_ctx = ssl.create_default_context()

    if args.data:
        data_path = args.data
    elif args.ood:
        data_path = f"{DATA_DIR}/test_ood.jsonl"
    else:
        data_path = f"{DATA_DIR}/test.jsonl"

    examples = build_examples(data_path, args.n, args.seed)
    total = len(examples)

    print(f"Loaded {total} examples from {data_path}")
    print(f"Endpoint: {args.url}")
    print(f"Concurrency: {args.concurrency}, seed: {args.seed}")
    print()

    printer = ResultPrinter(total, use_colour)
    results = []

    t0 = time.monotonic()

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {executor.submit(classify_safe, args.url, ex, ssl_ctx): ex for ex in examples}
        for future in as_completed(futures):
            result = future.result()
            printer.print_result(result)
            results.append(result)

            if printer.errors > total * ERROR_THRESHOLD:
                print()
                print(
                    red(
                        f"ERROR: >{ERROR_THRESHOLD * 100:.0f}% of requests failed"
                        f" ({printer.errors}/{total}). Aborting.",
                        use_colour,
                    )
                )
                sys.exit(1)

    elapsed = time.monotonic() - t0

    # Clear running line
    sys.stdout.write("\r" + " " * 80 + "\r")
    sys.stdout.flush()

    exit_code = print_report(
        results, printer.errors, total, args.concurrency, elapsed, use_colour
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
