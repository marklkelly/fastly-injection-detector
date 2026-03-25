#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
if command -v uv &>/dev/null; then
    uv run python ml/training/train_cls.py --config ml/training/config/model.yaml "$@"
else
    python3 ml/training/train_cls.py --config ml/training/config/model.yaml "$@"
fi
