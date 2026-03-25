"""
eval_ood.py — Evaluate a trained model checkpoint against any labeled JSONL file.

Intended for OOD (out-of-distribution) test sets such as test_ood.jsonl.

Usage:
    python ml/training/eval_ood.py \
        --model-dir ml/models/bert-tiny-pi-v2 \
        --eval-path ml/data/versions/pi_mix_v2/test_ood.jsonl \
        --output-path ml/models/bert-tiny-pi-v2/ood_eval.json \
        --prior 0.02
"""

import sys
import os
import json
import argparse

# Allow importing eval_utils from ml/data/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
from eval_utils import evaluate_at_prior

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import average_precision_score, roc_auc_score


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained model on a labeled JSONL file (OOD eval)")
    p.add_argument("--model-dir", type=str, required=True,
                   help="Path to trained model directory (contains tokenizer, model weights, calibrated_thresholds.json)")
    p.add_argument("--eval-path", type=str, required=True,
                   help="Path to eval JSONL file (fields: text, label; label 0=SAFE 1=INJECTION)")
    p.add_argument("--output-path", type=str, required=True,
                   help="Path to write the output JSON")
    p.add_argument("--prior", type=float, default=0.02,
                   help="Real-world prior (prevalence of positives) for adjusted metrics (default: 0.02)")
    p.add_argument("--batch-size", type=int, default=64,
                   help="Inference batch size (default: 64)")
    p.add_argument("--max-length", type=int, default=128,
                   help="Max sequence length for tokenization (default: 128)")
    return p.parse_args()


def select_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_jsonl(path):
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj["text"]
            raw_label = obj["label"]
            if isinstance(raw_label, str):
                up = raw_label.strip().upper()
                if up in ("INJECTION", "1"):
                    label = 1
                elif up in ("SAFE", "0"):
                    label = 0
                else:
                    raise ValueError(f"Unknown label value: {raw_label!r}")
            else:
                label = int(raw_label)
            examples.append((text, label))
    return examples


def run_inference(model, tokenizer, texts, device, batch_size, max_length):
    """Run batched inference and return softmax probabilities for class 1."""
    model.eval()
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        enc = tokenizer(
            batch_texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        logits = out.logits.float().cpu().numpy()
        z = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(z)
        sm = e / e.sum(axis=1, keepdims=True)
        all_probs.extend(sm[:, 1].tolist())
    return all_probs


def metrics_at_threshold(y_true, scores, threshold):
    """Compute balanced (unweighted) F1, precision, recall, FPR, TPR at a threshold."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    pred = (scores >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    n_pos = tp + fn
    n_neg = fp + tn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    tpr = tp / n_pos if n_pos > 0 else 0.0
    fpr = fp / n_neg if n_neg > 0 else 0.0
    return {
        "threshold": float(threshold),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "fpr": float(fpr),
        "tpr": float(tpr),
    }


def main():
    args = parse_args()

    # Load thresholds
    thr_path = os.path.join(args.model_dir, "calibrated_thresholds.json")
    with open(thr_path, "r") as f:
        thresholds = json.load(f)
    t_block = float(thresholds["T_block_at_1pct_FPR"])
    t_review = float(thresholds["T_review_lower_at_2pct_FPR"])

    # Load eval data
    examples = load_jsonl(args.eval_path)
    texts = [e[0] for e in examples]
    y_true = [e[1] for e in examples]
    n_examples = len(examples)
    print(f"Loaded {n_examples} examples from {args.eval_path}")

    # Load model and tokenizer
    device = select_device()
    print(f"Using device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model = model.to(device)

    # Run inference
    scores = run_inference(model, tokenizer, texts, device, args.batch_size, args.max_length)

    y_true_arr = np.asarray(y_true)
    scores_arr = np.asarray(scores)

    # AUC metrics
    auc_pr = float(average_precision_score(y_true_arr, scores_arr))
    auc_roc = float(roc_auc_score(y_true_arr, scores_arr))

    # Balanced metrics at each threshold
    block_balanced = metrics_at_threshold(y_true, scores, t_block)
    review_balanced = metrics_at_threshold(y_true, scores, t_review)

    # Prevalence-adjusted metrics
    block_adjusted = evaluate_at_prior(y_true, scores, t_block, prior=args.prior)
    review_adjusted = evaluate_at_prior(y_true, scores, t_review, prior=args.prior)

    result = {
        "eval_path": args.eval_path,
        "model_dir": args.model_dir,
        "prior": args.prior,
        "threshold_source": "calibrated_thresholds.json",
        "n_examples": n_examples,
        "balanced": {
            "auc_pr": auc_pr,
            "auc_roc": auc_roc,
            "at_block_threshold": block_balanced,
            "at_review_threshold": review_balanced,
        },
        "estimated_at_prior": {
            "at_block_threshold": block_adjusted,
            "at_review_threshold": review_adjusted,
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results written to {args.output_path}")

    # Human-readable summary
    print("\n=== OOD Evaluation Summary ===")
    print(f"  Examples:         {n_examples}")
    print(f"  AUC-PR:           {auc_pr:.4f}")
    print(f"  AUC-ROC:          {auc_roc:.4f}")
    print(f"\n  Block threshold ({t_block:.4f}):")
    print(f"    F1={block_balanced['f1']:.4f}  Prec={block_balanced['precision']:.4f}  "
          f"Rec={block_balanced['recall']:.4f}  FPR={block_balanced['fpr']:.4f}")
    print(f"    Est. PPV@{args.prior:.0%}: {block_adjusted['estimated_ppv']:.4f}  "
          f"Est. F1@{args.prior:.0%}: {block_adjusted['estimated_f1']:.4f}")
    print(f"\n  Review threshold ({t_review:.4f}):")
    print(f"    F1={review_balanced['f1']:.4f}  Prec={review_balanced['precision']:.4f}  "
          f"Rec={review_balanced['recall']:.4f}  FPR={review_balanced['fpr']:.4f}")
    print(f"    Est. PPV@{args.prior:.0%}: {review_adjusted['estimated_ppv']:.4f}  "
          f"Est. F1@{args.prior:.0%}: {review_adjusted['estimated_f1']:.4f}")


if __name__ == "__main__":
    main()
