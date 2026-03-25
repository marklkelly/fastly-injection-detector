#!/usr/bin/env python3
"""
Build a unified prompt-injection dataset from multiple Hugging Face sources.

Outputs:
  out_dir/
    train.jsonl
    val.jsonl
    test.jsonl
    labels.json              # ["SAFE","INJECTION"]
    manifest.json            # stats per source and overall
    combined.arrow/          # (optional) HF Dataset saved-to-disk
"""

import argparse, json, os, random, re, sys, hashlib, math
from dataclasses import dataclass, asdict
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple, Optional

from datasets import load_dataset, Dataset, concatenate_datasets, Value

try:
    import yaml as _yaml
except ImportError:
    _yaml = None

LABEL_SAFE = 0
LABEL_INJ = 1
LABELS = ["SAFE", "INJECTION"]

# ---------- utils

def normalise_text(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.replace("\u200b", " ")  # zero-width space
    s = re.sub(r"\s+", " ", s).strip()
    return s

def hash_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def pick_text_column(cols: Dict[str, Any]) -> str:
    # prefer common names
    for k in ["text", "prompt", "input", "question"]:
        if k in cols:
            return k
    # fall back to the first string-ish column
    for k, v in cols.items():
        if v.dtype == "string":
            return k
    # last resort: first column
    return list(cols.keys())[0]

def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

@dataclass
class SourceSpec:
    repo: str
    config: str = None
    split: str = None
    gated: bool = False
    # name override for manifest
    name: str = None

# ---------- loaders per dataset (robust to small schema variations)

def load_jayavibhav_prompt_injection(limit:int=None, revision:str=None) -> Dataset:
    """
    HF: jayavibhav/prompt-injection
    Columns: text (str), label (int64: 0 safe, 1 injection) — verified via viewer.
    """
    ds = load_dataset("jayavibhav/prompt-injection", revision=revision)
    use = ds["train"].flatten_indices().shuffle(seed=42)
    if limit:
        use = use.select(range(min(limit, len(use))))
    # normalise
    def map_fn(ex):
        return {
            "text": normalise_text(ex["text"]),
            "label": int(ex["label"]),
            "source": "jayavibhav/prompt-injection",
        }
    return use.map(map_fn, remove_columns=[c for c in use.column_names if c not in []])

def load_xTRam1_safe_guard(limit: int = None, revision: str = None) -> Dataset:
    """
    HF: xTRam1/safe-guard-prompt-injection
    Schema: text (string), label (int64).
    Hardcoded mapping (verified via manual inspection 2026-03-12): 0=SAFE, 1=INJECTION.
    Fails loudly if schema changes rather than falling back to heuristics.
    """
    ds = load_dataset("xTRam1/safe-guard-prompt-injection", revision=revision)
    use = ds["train"].flatten_indices().shuffle(seed=42)
    if limit:
        use = use.select(range(min(limit, len(use))))

    label_feature = use.features.get("label")
    if getattr(label_feature, "dtype", None) != "int64":
        raise ValueError(
            f"xTRam1 schema changed: expected int64 label, got {label_feature}. "
            "Inspect the dataset and update the hardcoded mapping."
        )
    unique_labels = {int(v) for v in use.unique("label")}
    if unique_labels != {0, 1}:
        raise ValueError(
            f"xTRam1 label values changed: expected {{0, 1}}, got {sorted(unique_labels)}."
        )

    inj_label = 1  # observed via manual inspection 2026-03-12

    def map_fn(ex):
        y = LABEL_INJ if int(ex["label"]) == inj_label else LABEL_SAFE
        return {"text": normalise_text(ex["text"]), "label": y, "source": "xTRam1/safe-guard-prompt-injection"}

    return use.map(map_fn, remove_columns=[c for c in use.column_names if c not in []])

def load_rubend18_jailbreak(limit: int = None, revision: str = None) -> Dataset:
    """
    HF: rubend18/ChatGPT-Jailbreak-Prompts — all rows are INJECTION.
    Used as fallback when Harelix is unavailable.
    """
    ds = load_dataset("rubend18/ChatGPT-Jailbreak-Prompts", revision=revision)
    use = (ds["train"] if "train" in ds else list(ds.values())[0]).flatten_indices().shuffle(seed=42)
    if limit:
        use = use.select(range(min(limit, len(use))))
    text_col = pick_text_column(use.features)

    def map_fn(ex):
        text = ex.get("text") or ex.get("act") or ex.get(text_col) or ""
        return {"text": normalise_text(str(text)), "label": LABEL_INJ, "source": "rubend18/ChatGPT-Jailbreak-Prompts"}

    return use.map(map_fn, remove_columns=[c for c in use.column_names if c not in []])


def load_neuralchemy_prompt_injection(limit: int = None, revision: str = None) -> Dataset:
    """
    HF: neuralchemy/Prompt-injection-dataset
    Columns: text (str), label (int32: 0=SAFE, 1=INJECTION), category, severity, etc.
    All three splits (train/validation/test) are concatenated.
    Schema verified 2026-03-12.
    """
    ds = load_dataset("neuralchemy/Prompt-injection-dataset", revision=revision)
    use = concatenate_datasets(list(ds.values())).flatten_indices().shuffle(seed=42)
    if limit:
        use = use.select(range(min(limit, len(use))))

    unique_labels = {int(v) for v in use.unique("label")}
    if unique_labels != {0, 1}:
        raise ValueError(
            f"neuralchemy label values changed: expected {{0, 1}}, got {sorted(unique_labels)}."
        )

    def map_fn(ex):
        return {
            "text": normalise_text(ex["text"]),
            "label": int(ex["label"]),
            "source": "neuralchemy/Prompt-injection-dataset",
        }
    result = use.map(map_fn, remove_columns=[c for c in use.column_names if c not in []])
    return result.cast_column("label", Value("int64"))


def load_wambosec_subtle(limit: int = None, revision: str = None) -> Dataset:
    """
    HF: wambosec/prompt-injections-subtle
    Columns: prompt (str), label (int64: 0=SAFE, 1=INJECTION), is_malicious, category, etc.
    Text is in 'prompt' column. Both splits (train/test) are concatenated.
    Schema verified 2026-03-12.
    """
    ds = load_dataset("wambosec/prompt-injections-subtle", revision=revision)
    use = concatenate_datasets(list(ds.values())).flatten_indices().shuffle(seed=42)
    if limit:
        use = use.select(range(min(limit, len(use))))

    unique_labels = {int(v) for v in use.unique("label")}
    if unique_labels != {0, 1}:
        raise ValueError(
            f"wambosec label values changed: expected {{0, 1}}, got {sorted(unique_labels)}."
        )

    def map_fn(ex):
        return {
            "text": normalise_text(ex["prompt"]),
            "label": int(ex["label"]),
            "source": "wambosec/prompt-injections-subtle",
        }
    return use.map(map_fn, remove_columns=[c for c in use.column_names if c not in []])


def load_harelix_or_fallback(cap: int = None, harelix_revision: str = None, fallback_revision: str = None) -> Tuple[Optional[Dataset], bool]:
    """
    Try Harelix/Prompt-Injection-Mixed-Techniques-2024; fall back to
    rubend18/ChatGPT-Jailbreak-Prompts at double cap (max 10000) on any load error.
    Raises ValueError if Harelix loads but produces zero INJECTION rows.
    Returns (dataset, harelix_succeeded).
    """
    fallback_cap = min((cap or 10_000) * 2, 10_000)

    try:
        ds = load_dataset("Harelix/Prompt-Injection-Mixed-Techniques-2024", revision=harelix_revision)
        use = (ds["train"] if "train" in ds else list(ds.values())[0]).flatten_indices().shuffle(seed=42)
    except Exception as e:
        print(
            f"[warn] Harelix unavailable ({type(e).__name__}: {str(e)[:80]}). "
            f"Falling back to rubend18/ChatGPT-Jailbreak-Prompts (cap={fallback_cap}).",
            file=sys.stderr,
        )
        return load_rubend18_jailbreak(limit=fallback_cap, revision=fallback_revision), False

    if cap:
        use = use.select(range(min(cap, len(use))))

    def map_fn(ex):
        text = ex.get("input") or ex.get("text") or ""
        lab = ex.get("label") or ""
        y = LABEL_INJ if str(lab).lower().strip() in {"malicious", "jailbreak", "attack"} else LABEL_SAFE
        return {"text": normalise_text(text), "label": y, "source": "Harelix/Prompt-Injection-Mixed-Techniques-2024"}

    mapped = use.map(map_fn, remove_columns=[c for c in use.column_names if c not in []])

    inj_count = sum(1 for y in mapped["label"] if y == LABEL_INJ)
    if inj_count == 0:
        raw_labels = sorted({str(v) for v in use["label"][: min(100, len(use))]})
        raise ValueError(
            f"Harelix/Prompt-Injection-Mixed-Techniques-2024 loaded but produced 0 INJECTION rows. "
            f"Observed raw labels (sample): {raw_labels}"
        )

    return mapped, True

def load_darkknight25_prompt_benign(limit:int=None, revision:str=None) -> Dataset:
    """
    HF: darkknight25/Prompt_Injection_Benign_Prompt_Dataset
    Schema preview shows fields like id, prompt, label in {malicious, benign}.
    """
    ds = load_dataset("darkknight25/Prompt_Injection_Benign_Prompt_Dataset", revision=revision)
    use = (ds["train"] if "train" in ds else list(ds.values())[0]).flatten_indices().shuffle(seed=42)
    if limit:
        use = use.select(range(min(limit, len(use))))
    # find text column
    text_col = pick_text_column(use.features)
    def map_fn(ex):
        txt = ex.get("prompt") or ex.get(text_col) or ""
        lab = ex.get("label") or ""
        y = LABEL_INJ if str(lab).lower().strip() in {"malicious","injection","attack","jailbreak"} else LABEL_SAFE
        return {"text": normalise_text(txt), "label": y, "source": "darkknight25/Prompt_Injection_Benign_Prompt_Dataset"}
    return use.map(map_fn, remove_columns=[c for c in use.column_names if c not in []])

def load_wildjailbreak(include: bool, limit: int = None, revision: str = None, include_adversarial_harmful: bool = False) -> Dataset:
    """
    HF (gated): allenai/wildjailbreak
    We map:
      data_type in {vanilla_harmful, adversarial_harmful} → INJECTION
                    {vanilla_benign, adversarial_benign} → SAFE
    Text = adversarial if non-empty else vanilla
    When include_adversarial_harmful=False (default), rows with data_type==adversarial_harmful
    are filtered out before limit selection (pending manual validation).
    """
    if not include:
        return None
    try:
        ds = load_dataset("allenai/wildjailbreak", "train", revision=revision, keep_default_na=False)
    except Exception as e:
        print("[warn] Could not load allenai/wildjailbreak (accept terms + login to include). Skipping.", file=sys.stderr)
        return None
    use = (ds["train"] if "train" in ds else list(ds.values())[0]).flatten_indices().shuffle(seed=42)
    if not include_adversarial_harmful:
        use = use.filter(lambda ex: (ex.get("data_type") or "").lower() != "adversarial_harmful")
    if limit:
        use = use.select(range(min(limit, len(use))))
    def map_fn(ex):
        text = ex.get("adversarial") or ex.get("vanilla") or ""
        dt = (ex.get("data_type") or "").lower()
        y = LABEL_INJ if ("harmful" in dt) else LABEL_SAFE
        return {"text": normalise_text(text), "label": y, "source": "allenai/wildjailbreak"}
    keep_cols = [c for c in use.column_names if c not in []]
    return use.map(map_fn, remove_columns=keep_cols)


def load_deepset_prompt_injections(limit: int = None, revision: str = None) -> Dataset:
    """
    HF: deepset/prompt-injections
    OOD-only source — never used in train/val/test splits.
    Schema: text (str), label (ClassLabel: 0=SAFE, 1=INJECTION or similar).
    """
    ds = load_dataset("deepset/prompt-injections", revision=revision)
    split = ds["train"] if "train" in ds else list(ds.values())[0]
    use = split.flatten_indices().shuffle(seed=42)
    if limit:
        use = use.select(range(min(limit, len(use))))

    label_feature = use.features.get("label")
    raw_labels = use.unique("label")
    print(f"[deepset] Observed raw label values: {sorted(raw_labels)}", file=sys.stderr)

    def map_fn(ex):
        raw = ex["label"]
        if isinstance(raw, str):
            y = LABEL_INJ if any(k in raw.lower() for k in ["inject", "attack", "malicious", "unsafe"]) else LABEL_SAFE
        else:
            y = int(raw)
        return {"text": normalise_text(ex["text"]), "label": y, "source": "deepset/prompt-injections"}

    mapped = use.map(map_fn, remove_columns=[c for c in use.column_names if c not in []])

    inj_count = sum(1 for y in mapped["label"] if y == LABEL_INJ)
    safe_count = sum(1 for y in mapped["label"] if y == LABEL_SAFE)
    if inj_count == 0 or safe_count == 0:
        raise ValueError(
            f"deepset/prompt-injections mapping produced {inj_count} INJECTION, {safe_count} SAFE rows. "
            f"Observed raw labels: {sorted(raw_labels)}. Inspect and fix mapping."
        )
    print(f"[deepset] Mapped: {safe_count} SAFE, {inj_count} INJECTION", file=sys.stderr)
    return mapped


def load_hackaprompt(limit: int = None, revision: str = None):
    """
    HF: hackaprompt/hackaprompt-dataset (gated — requires HF terms acceptance).
    All rows are INJECTION (challenges to bypass an LLM prompt).
    Returns None on load failure (not a hard error — gated source).
    """
    try:
        ds = load_dataset("hackaprompt/hackaprompt-dataset", revision=revision)
    except Exception as e:
        print(
            f"[info] hackaprompt/hackaprompt-dataset unavailable ({type(e).__name__}). "
            "Accept terms at https://huggingface.co/datasets/hackaprompt/hackaprompt-dataset to enable.",
            file=sys.stderr,
        )
        return None
    split = ds["train"] if "train" in ds else list(ds.values())[0]
    use = split.flatten_indices().shuffle(seed=42)
    if limit:
        use = use.select(range(min(limit, len(use))))
    text_col = pick_text_column(use.features)
    def map_fn(ex):
        text = ex.get("user_input") or ex.get("prompt") or ex.get(text_col) or ""
        return {"text": normalise_text(str(text)), "label": LABEL_INJ, "source": "hackaprompt/hackaprompt-dataset"}
    return use.map(map_fn, remove_columns=[c for c in use.column_names if c not in []])


def load_markush1_injection(limit: int = None, revision: str = None, pending_license_review: bool = False):
    """
    HF: markush1/LLM-Injection-Dataset (pending license review).
    Skip with warning if pending_license_review=True.
    """
    if pending_license_review:
        print(
            "[warn] markush1/LLM-Injection-Dataset is flagged as pending_license_review. "
            "Skipping until human review completes.",
            file=sys.stderr,
        )
        return None
    try:
        ds = load_dataset("markush1/LLM-Injection-Dataset", revision=revision)
    except Exception as e:
        print(f"[warn] markush1/LLM-Injection-Dataset unavailable ({type(e).__name__}: {str(e)[:80]}). Skipping.", file=sys.stderr)
        return None
    split = ds["train"] if "train" in ds else list(ds.values())[0]
    use = split.flatten_indices().shuffle(seed=42)
    if limit:
        use = use.select(range(min(limit, len(use))))

    raw_labels = use.unique("label") if "label" in use.column_names else []
    print(f"[markush1] Observed raw label values: {sorted(str(x) for x in raw_labels)}", file=sys.stderr)
    text_col = pick_text_column(use.features)

    def map_fn(ex):
        txt = ex.get("text") or ex.get("prompt") or ex.get(text_col) or ""
        raw = ex.get("label", "")
        if isinstance(raw, str):
            y = LABEL_INJ if any(k in raw.lower() for k in ["inject", "attack", "malicious", "jailbreak"]) else LABEL_SAFE
        else:
            y = LABEL_INJ if int(raw) == 1 else LABEL_SAFE
        return {"text": normalise_text(str(txt)), "label": y, "source": "markush1/LLM-Injection-Dataset"}

    return use.map(map_fn, remove_columns=[c for c in use.column_names if c not in []])

# ---------- main combine / dedupe / balance / split

def stratified_split(rows: List[Dict[str,Any]], seed:int, ratios=(0.8,0.1,0.1)):
    random.seed(seed)
    by = defaultdict(list)
    for r in rows:
        by[r["label"]].append(r)
    for k in by:
        random.shuffle(by[k])
    train, val, test = [], [], []
    for k, items in by.items():
        n = len(items)
        n_train = int(n*ratios[0])
        n_val   = int(n*ratios[1])
        train.extend(items[:n_train])
        val.extend(items[n_train:n_train+n_val])
        test.extend(items[n_train+n_val:])
    random.shuffle(train); random.shuffle(val); random.shuffle(test)
    return train, val, test


def shingles_for_dedup(text: str) -> set:
    norm = normalise_text(text).lower()
    tokens = norm.split()
    if len(tokens) >= 3:
        return {" ".join(tokens[i:i+3]) for i in range(len(tokens) - 2)}
    if len(norm) >= 5:
        return {norm[i:i+5] for i in range(len(norm) - 4)}
    return {norm}


def build_minhash(text: str, num_perm: int = 128):
    from datasketch import MinHash
    m = MinHash(num_perm=num_perm)
    for shingle in sorted(shingles_for_dedup(text)):
        m.update(shingle.encode("utf-8"))
    return m


def build_minhash_clusters(rows, num_perm: int = 128, threshold: float = 0.7):
    """
    Deduplicates rows by MinHash near-duplicate clustering.
    Keeps only the first row encountered per cluster (deterministic representative).
    Returns (deduplicated_rows, num_clusters, rows_removed).
    """
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        print("[warn] datasketch not installed. MinHash dedup skipped. Install with: pip install datasketch", file=sys.stderr)
        result = [{**row, "cluster_id": i} for i, row in enumerate(rows)]
        return result, len(rows), 0

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    cluster_map = {}
    next_cluster = 0

    for i, row in enumerate(rows):
        if i > 0 and i % 10000 == 0:
            print(f"[minhash] Processed {i}/{len(rows)} rows …", file=sys.stderr)
        m = build_minhash(row["text"], num_perm=num_perm)
        neighbors = lsh.query(m)
        if neighbors:
            cid = cluster_map[int(neighbors[0])]
        else:
            cid = next_cluster
            next_cluster += 1
        cluster_map[i] = cid
        try:
            lsh.insert(str(i), m)
        except ValueError:
            pass  # duplicate exact hash

    # Keep only the first row seen per cluster
    rows_before = len(rows)
    seen_clusters: set = set()
    deduped = []
    for i, row in enumerate(rows):
        cid = cluster_map[i]
        if cid not in seen_clusters:
            seen_clusters.add(cid)
            deduped.append({**row, "cluster_id": cid})

    rows_removed = rows_before - len(deduped)
    print(
        f"[minhash] Removed {rows_removed} near-duplicate rows ({rows_before} → {rows_before - rows_removed})",
        file=sys.stderr,
    )
    return deduped, next_cluster, rows_removed


def load_edge_case_corpus(limit: int = None, revision: str = None) -> "Dataset":
    """Load the hand-curated edge case corpus from ml/data/manual/edge_case_corpus_v1.jsonl."""
    if Dataset is None:
        raise RuntimeError("'datasets' package is required. Run: pip install datasets")
    path = os.path.join(os.path.dirname(__file__), "manual", "edge_case_corpus_v1.jsonl")
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                rows.append({
                    "text": normalise_text(r["text"]),
                    "label": int(r["label"]),
                    "source": "local/edge_case_corpus_v1",
                })
    if limit:
        rows = rows[:limit]
    return Dataset.from_list(rows)


def cluster_stratified_split(rows, seed=42, ratios=(0.8, 0.1, 0.1)):
    """
    Split rows by cluster_id within each (source, label) bucket.
    Keeps near-duplicate clusters entirely in one split.
    Minimum-size rule: buckets with <10 rows go to train only.
    """
    import random as _random
    _random.seed(seed)
    by_src_label = defaultdict(list)
    for row in rows:
        by_src_label[(row["source"], row["label"])].append(row)

    train, val, test = [], [], []

    for (source, label), items in by_src_label.items():
        if len(items) < 10:
            train.extend(items)
            continue

        clusters = defaultdict(list)
        for item in items:
            clusters[item.get("cluster_id", hash(item["text"]) % 100000)].append(item)

        cluster_items = list(clusters.items())
        _random.shuffle(cluster_items)

        total = len(items)
        n_train = int(total * ratios[0])
        n_val = int(total * ratios[1])

        running = 0
        for _, group in cluster_items:
            if running < n_train:
                train.extend(group)
            elif running < n_train + n_val:
                val.extend(group)
            else:
                test.extend(group)
            running += len(group)

    _random.shuffle(train)
    _random.shuffle(val)
    _random.shuffle(test)
    return train, val, test

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recipe", metavar="PATH",
                    help="YAML recipe file. Provides defaults for seed, min_chars, max_chars, "
                         "balance, and per-source caps. Explicit CLI flags override recipe values.")
    ap.add_argument("--output", metavar="PATH",
                    help="Output directory (alias for --out-dir).")
    ap.add_argument("--out-dir", default="data/pi_mix_v1",
                    help="Output directory (default: data/pi_mix_v1). --output takes precedence.")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--include-wildjailbreak", action="store_true",
                    help="Include allenai/wildjailbreak (requires accepting terms on HF). "
                         "Superseded by recipe sources list when --recipe is given.")
    ap.add_argument("--include-wjb-harmful", action="store_true",
                    help="Include adversarial_harmful rows from allenai/wildjailbreak (requires manual validation).")
    ap.add_argument("--limit-per-source", type=int, default=0,
                    help="Override cap for every source after shuffle (0 = use recipe/default caps).")
    ap.add_argument("--min-chars", type=int)
    ap.add_argument("--max-chars", type=int)
    ap.add_argument("--balance", choices=["off", "downsample", "upsample"])
    ap.add_argument("--save-hf", action="store_true", help="Also save the combined Dataset to disk.")
    ap.add_argument("--skip-minhash", action="store_true",
                    help="Skip MinHash deduplication (use exact dedup only). Faster but less thorough.")
    args = ap.parse_args()

    # Load recipe -------------------------------------------------------
    recipe: Dict[str, Any] = {}
    recipe_path_abs: Optional[str] = None
    recipe_digest: Optional[str] = None
    if args.recipe:
        if _yaml is None:
            print("--recipe requires the 'pyyaml' package:\n  pip install pyyaml", file=sys.stderr)
            sys.exit(1)
        recipe_path_abs = os.path.abspath(args.recipe)
        with open(recipe_path_abs) as _f:
            recipe = _yaml.safe_load(_f) or {}
        with open(recipe_path_abs, "rb") as _f:
            recipe_digest = hashlib.sha256(_f.read()).hexdigest()

    # Merge: CLI > recipe > hardcoded defaults --------------------------
    out_dir = args.output or args.out_dir
    seed = args.seed if args.seed is not None else recipe.get("seed", 42)
    min_chars = args.min_chars if args.min_chars is not None else recipe.get("min_chars", 8)
    max_chars = args.max_chars if args.max_chars is not None else recipe.get("max_chars", 4000)
    balance = args.balance or recipe.get("balance", "downsample")

    # Build per-source cap lookup from recipe ---------------------------
    recipe_source_cfg: Dict[str, Dict] = {}
    for src in recipe.get("sources", []):
        if isinstance(src, dict) and "repo" in src:
            recipe_source_cfg[src["repo"]] = src

    def get_cap(repo: str, hardcoded_default: int) -> int:
        """Return effective cap: --limit-per-source > recipe cap > hardcoded default."""
        if args.limit_per_source:
            return args.limit_per_source
        cfg = recipe_source_cfg.get(repo, {})
        return cfg.get("cap") or hardcoded_default

    def get_rev(repo: str) -> Optional[str]:
        """Return pinned revision for repo from recipe, or None if not pinned."""
        return (recipe_source_cfg.get(repo) or {}).get("revision")

    os.makedirs(out_dir, exist_ok=True)

    # Determine which sources to load -----------------------------------
    # When a recipe is present, load only sources listed in it.
    # Without a recipe, fall back to the original hardcoded sequence.
    datasets_list = []
    source_provenance: List[Dict[str, Any]] = []
    ood_rows: List[Dict[str, Any]] = []
    ood_source_repos: set = set()

    _LOADER_MAP = {
        "jayavibhav/prompt-injection": lambda lim, rev: load_jayavibhav_prompt_injection(limit=lim, revision=rev),
        "neuralchemy/Prompt-injection-dataset": lambda lim, rev: load_neuralchemy_prompt_injection(limit=lim, revision=rev),
        "wambosec/prompt-injections-subtle": lambda lim, rev: load_wambosec_subtle(limit=lim, revision=rev),
        "xTRam1/safe-guard-prompt-injection": lambda lim, rev: load_xTRam1_safe_guard(limit=lim, revision=rev),
        "darkknight25/Prompt_Injection_Benign_Prompt_Dataset": lambda lim, rev: load_darkknight25_prompt_benign(limit=lim, revision=rev),
        "allenai/wildjailbreak": lambda lim, rev: load_wildjailbreak(include=True, limit=lim, revision=rev, include_adversarial_harmful=args.include_wjb_harmful),
        "deepset/prompt-injections": lambda lim, rev: load_deepset_prompt_injections(limit=lim, revision=rev),
        "hackaprompt/hackaprompt-dataset": lambda lim, rev: load_hackaprompt(limit=lim, revision=rev),
        "markush1/LLM-Injection-Dataset": lambda lim, rev, plr=False: load_markush1_injection(limit=lim, revision=rev, pending_license_review=plr),
        "local/edge_case_corpus_v1": lambda lim, rev: load_edge_case_corpus(limit=lim, revision=rev),
    }

    if recipe.get("sources"):
        for src_cfg in recipe["sources"]:
            if not isinstance(src_cfg, dict) or "repo" not in src_cfg:
                continue
            repo = src_cfg["repo"]
            rev = get_rev(repo)
            is_ood = src_cfg.get("ood_eval") is True
            pending_lr = src_cfg.get("pending_license_review", False)

            # Handle gated sources (non-OOD, non-markush1)
            if src_cfg.get("gated") and not is_ood and repo not in ("markush1/LLM-Injection-Dataset", "hackaprompt/hackaprompt-dataset"):
                print(f"[info] Skipping {repo} (gated; add access then re-run).", file=sys.stderr)
                source_provenance.append({
                    "repo": repo,
                    "revision": rev,
                    "status": "skipped",
                    "reason": "gated source not included",
                })
                continue

            cap_val = get_cap(repo, src_cfg.get("cap") or 0) or None

            if repo == "Harelix/Prompt-Injection-Mixed-Techniques-2024":
                harelix_rev = rev
                fallback_rev = src_cfg.get("fallback_revision")
                try:
                    d, harelix_ok = load_harelix_or_fallback(
                        cap=cap_val,
                        harelix_revision=harelix_rev,
                        fallback_revision=fallback_rev,
                    )
                    if d is not None:
                        datasets_list.append(d)
                        if harelix_ok:
                            source_provenance.append({
                                "repo": repo,
                                "revision": harelix_rev,
                                "status": "used",
                                "rows_loaded": len(d),
                            })
                        else:
                            source_provenance.append({
                                "repo": repo,
                                "revision": harelix_rev,
                                "status": "replaced",
                                "fallback_repo": "rubend18/ChatGPT-Jailbreak-Prompts",
                                "fallback_revision": fallback_rev,
                                "rows_loaded": len(d),
                            })
                except ValueError:
                    raise
                except Exception as e:
                    print(f"[warn] skip {repo}: {e}", file=sys.stderr)
                continue

            # OOD source — load into datasets_list (global dedup runs before routing)
            if is_ood:
                ood_source_repos.add(repo)
                ldr = _LOADER_MAP.get(repo)
                if ldr is None:
                    print(f"[warn] No loader for OOD source {repo}. Skipping.", file=sys.stderr)
                    continue
                try:
                    d = ldr(cap_val, rev)
                    if d is not None:
                        datasets_list.append(d)
                        source_provenance.append({
                            "repo": repo,
                            "revision": rev,
                            "status": "used",
                            "ood_eval": True,
                            "rows_loaded": len(d),
                        })
                        print(f"[info] OOD source {repo}: {len(d)} rows loaded (dedup before routing)", file=sys.stderr)
                except ValueError:
                    raise
                except Exception as e:
                    print(f"[warn] skip OOD {repo}: {e}", file=sys.stderr)
                continue

            # markush1 — pass pending_license_review flag
            if repo == "markush1/LLM-Injection-Dataset":
                if pending_lr:
                    print(f"[info] Skipping {repo} (pending_license_review).", file=sys.stderr)
                    source_provenance.append({
                        "repo": repo,
                        "revision": rev,
                        "status": "skipped",
                        "reason": "pending_license_review",
                    })
                    continue
                ldr = _LOADER_MAP.get(repo)
                try:
                    d = ldr(cap_val, rev, False)
                    if d is not None:
                        datasets_list.append(d)
                        source_provenance.append({
                            "repo": repo,
                            "revision": rev,
                            "status": "used",
                            "rows_loaded": len(d),
                        })
                except ValueError:
                    raise
                except Exception as e:
                    print(f"[warn] skip {repo}: {e}", file=sys.stderr)
                continue

            ldr = _LOADER_MAP.get(repo)
            if ldr is None:
                print(f"[warn] No loader implemented for {repo}. Skipping.", file=sys.stderr)
                continue
            try:
                d = ldr(cap_val, rev)
                if d is not None:
                    datasets_list.append(d)
                    source_provenance.append({
                        "repo": repo,
                        "revision": rev,
                        "status": "used",
                        "rows_loaded": len(d),
                    })
            except ValueError:
                raise  # schema/data validation errors are fatal
            except Exception as e:
                print(f"[warn] skip {repo}: {e}", file=sys.stderr)
    else:
        # No recipe: use original hardcoded sequence for backwards compat.
        for repo, default_cap in [
            ("jayavibhav/prompt-injection", 200_000),
            ("xTRam1/safe-guard-prompt-injection", 10_000),
            ("Harelix/Prompt-Injection-Mixed-Techniques-2024", 10_000),
            ("darkknight25/Prompt_Injection_Benign_Prompt_Dataset", 5_000),
        ]:
            try:
                if repo == "Harelix/Prompt-Injection-Mixed-Techniques-2024":
                    d, harelix_ok = load_harelix_or_fallback(cap=get_cap(repo, default_cap))
                    if d is not None:
                        datasets_list.append(d)
                        entry: Dict[str, Any] = {
                            "repo": repo,
                            "revision": None,
                            "status": "used" if harelix_ok else "replaced",
                            "rows_loaded": len(d),
                        }
                        if not harelix_ok:
                            entry["fallback_repo"] = "rubend18/ChatGPT-Jailbreak-Prompts"
                            entry["fallback_revision"] = None
                        source_provenance.append(entry)
                else:
                    d = _LOADER_MAP[repo](get_cap(repo, default_cap), None)
                    if d is not None:
                        datasets_list.append(d)
                        source_provenance.append({
                            "repo": repo,
                            "revision": None,
                            "status": "used",
                            "rows_loaded": len(d),
                        })
            except ValueError:
                raise  # schema/data validation errors are fatal
            except Exception as e:
                print(f"[warn] skip {repo}: {e}", file=sys.stderr)

        if args.include_wildjailbreak:
            wjb_rev = get_rev("allenai/wildjailbreak")
            d5 = load_wildjailbreak(include=True, limit=get_cap("allenai/wildjailbreak", 50_000), revision=wjb_rev)
            if d5 is not None:
                datasets_list.append(d5)
                source_provenance.append({
                    "repo": "allenai/wildjailbreak",
                    "revision": None,
                    "status": "used",
                    "rows_loaded": len(d5),
                })

    if not datasets_list:
        print("No datasets loaded. Check connectivity and/or gate acceptance.", file=sys.stderr)
        sys.exit(2)

    # Concatenate -------------------------------------------------------
    combined = concatenate_datasets(datasets_list)

    # Length filter -----------------------------------------------------
    def keep_len(ex):
        t = ex["text"]
        if not isinstance(t, str):
            return False
        return min_chars <= len(t) <= max_chars

    combined = combined.filter(keep_len)

    # Exact dedup -------------------------------------------------------
    print("[info] Deduplicating …")
    rows_before_exact = len(combined)
    hashes: set = set()
    keep_idx = []
    for i, t in enumerate(combined["text"]):
        h = hash_text(t.lower())
        if h in hashes:
            continue
        hashes.add(h)
        keep_idx.append(i)
    combined = combined.select(keep_idx)
    rows_after_exact = len(combined)
    print(f"[info] Exact dedup: {rows_before_exact} → {rows_after_exact} rows", file=sys.stderr)

    counts_before = Counter(int(y) for y in combined["label"])

    # Convert to rows list (include OOD; split happens after MinHash)
    all_rows = [
        {"text": t, "label": int(y), "source": s}
        for t, y, s in zip(combined["text"], combined["label"], combined["source"])
    ]
    random.seed(seed)
    random.shuffle(all_rows)

    # MinHash dedup (optional) — run globally across in-domain + OOD before routing
    minhash_num_perm = 128
    minhash_threshold = 0.7
    minhash_skipped = args.skip_minhash
    num_clusters = len(all_rows)
    minhash_removed = 0

    if not args.skip_minhash:
        dedup_cfg = recipe.get("dedup", {}) if recipe else {}
        minhash_num_perm = dedup_cfg.get("num_perm", 128)
        minhash_threshold = dedup_cfg.get("threshold", 0.7)
        print(f"[info] Running MinHash dedup (num_perm={minhash_num_perm}, threshold={minhash_threshold}) …", file=sys.stderr)
        all_rows, num_clusters, minhash_removed = build_minhash_clusters(all_rows, num_perm=minhash_num_perm, threshold=minhash_threshold)
        print(f"[info] MinHash: {len(all_rows)} rows in {num_clusters} clusters", file=sys.stderr)
    else:
        # Assign sequential cluster IDs so cluster_stratified_split still works
        all_rows = [{**r, "cluster_id": i} for i, r in enumerate(all_rows)]

    # Route OOD rows out; pass in-domain rows to balance + split
    if ood_source_repos:
        ood_rows = [r for r in all_rows if r["source"] in ood_source_repos]
        rows = [r for r in all_rows if r["source"] not in ood_source_repos]
        print(f"[info] After global dedup: routed {len(ood_rows)} OOD rows → test_ood.jsonl", file=sys.stderr)
    else:
        rows = all_rows

    # Balance (in-domain rows only) -------------------------------------
    if balance != "off":
        by: defaultdict = defaultdict(list)
        for r in rows:
            by[r["label"]].append(r)
        n0, n1 = len(by[LABEL_SAFE]), len(by[LABEL_INJ])
        if n0 == 0 or n1 == 0:
            print("[warn] Skipping balance — one class is empty.", file=sys.stderr)
        elif balance == "downsample":
            target = min(n0, n1)
            rows = random.sample(by[LABEL_SAFE], target) + random.sample(by[LABEL_INJ], target)
            random.shuffle(rows)
        else:  # upsample
            target = max(n0, n1)

            def up(lst):
                mul = math.ceil(target / len(lst))
                return (lst * mul)[:target]

            rows = up(by[LABEL_SAFE]) + up(by[LABEL_INJ])
            random.shuffle(rows)

    # Split -------------------------------------------------------------
    if not args.skip_minhash:
        train, val, test = cluster_stratified_split(rows, seed=seed, ratios=(0.8, 0.1, 0.1))
    else:
        train, val, test = stratified_split(rows, seed=seed, ratios=(0.8, 0.1, 0.1))

    # Persist -----------------------------------------------------------
    # Strip cluster_id from output rows
    def clean_row(r):
        return {"text": r["text"], "label": r["label"], "source": r["source"]}

    with open(os.path.join(out_dir, "labels.json"), "w") as f:
        json.dump(LABELS, f, indent=2)
    write_jsonl(os.path.join(out_dir, "train.jsonl"), [clean_row(r) for r in train])
    write_jsonl(os.path.join(out_dir, "val.jsonl"), [clean_row(r) for r in val])
    write_jsonl(os.path.join(out_dir, "test.jsonl"), [clean_row(r) for r in test])

    if ood_rows:
        write_jsonl(os.path.join(out_dir, "test_ood.jsonl"), [clean_row(r) for r in ood_rows])
        print(f"[info] Wrote {len(ood_rows)} OOD rows to test_ood.jsonl", file=sys.stderr)

    if args.save_hf:
        from datasets import DatasetDict
        dd = DatasetDict({
            "train": Dataset.from_list([clean_row(r) for r in train]),
            "validation": Dataset.from_list([clean_row(r) for r in val]),
            "test": Dataset.from_list([clean_row(r) for r in test]),
        })
        dd.save_to_disk(os.path.join(out_dir, "combined.arrow"))

    # Manifest ----------------------------------------------------------
    import datasets as _hf_datasets
    import datetime

    try:
        import datasketch as _datasketch
        datasketch_version = _datasketch.__version__
    except Exception:
        datasketch_version = "not_installed"

    exact_removed = rows_before_exact - rows_after_exact
    dataset_version = recipe.get("version", "unknown") if recipe else "unknown"

    manifest: Dict[str, Any] = {
        "version": dataset_version,
        "build_date": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "build_command": " ".join(sys.argv),
        "recipe_path": recipe_path_abs,
        "recipe_sha256": recipe_digest,
        "python_dependencies": {
            "datasets": _hf_datasets.__version__,
            "datasketch": datasketch_version,
        },
        "sources": source_provenance,
        "dedup": {
            "exact_removed": exact_removed,
            "minhash_removed": minhash_removed,
            "threshold": minhash_threshold,
            "num_perm": minhash_num_perm,
        },
        "final": {
            "train": len(train),
            "val": len(val),
            "test": len(test),
            "test_ood": len(ood_rows),
        },
    }

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[done] Wrote dataset to: {out_dir}")
    print("       Files: train.jsonl, val.jsonl, test.jsonl, labels.json, manifest.json")

if __name__ == "__main__":
    main()
