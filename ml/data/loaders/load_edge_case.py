import json
from pathlib import Path
from typing import List, Dict, Any

_DEFAULT_PATH = Path(__file__).parent.parent / "manual" / "edge_case_corpus_v1.jsonl"

def load_edge_cases(path: str = None) -> List[Dict[str, Any]]:
    """Load edge case corpus from JSONL file. Returns list of dicts with text, label, source, category."""
    p = Path(path) if path else _DEFAULT_PATH
    rows = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
