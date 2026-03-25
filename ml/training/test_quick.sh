#!/usr/bin/env bash
#
# Quick test to validate the config-backed training pipeline.
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  PYTHON_BIN="python3"
fi

echo "==========================================="
echo "Quick Pipeline Test"
echo "==========================================="
echo "This will test the training pipeline with:"
echo "  - model config: ml/training/config/model.yaml"
echo "  - 100 training samples"
echo "  - 50 validation samples"
echo "  - 2 training steps"
echo "  - Full evaluation and saving"
echo ""

OUTPUT_DIR="/tmp/quick_test_$(date +%s)"
mkdir -p "${OUTPUT_DIR}"

echo "Starting quick test..."
"${PYTHON_BIN}" "${REPO_ROOT}/ml/training/train_cls.py" \
  --config "${REPO_ROOT}/ml/training/config/model.yaml" \
  --quick_test \
  --output_dir "${OUTPUT_DIR}"

echo ""
echo "Checking outputs..."
echo "==========================================="

EXPECTED_FILES=(
  "config.json"
  "model.safetensors"
  "tokenizer_config.json"
  "tokenizer.json"
  "labels.json"
  "eval_metrics.json"
  "calibrated_thresholds.json"
  "model_card.json"
)

ALL_GOOD=true
for FILE in "${EXPECTED_FILES[@]}"; do
  if [ -f "${OUTPUT_DIR}/${FILE}" ]; then
    echo "✅ ${FILE}"
  else
    echo "❌ ${FILE} (missing)"
    ALL_GOOD=false
  fi
done

if [ -d "${OUTPUT_DIR}/edge_export" ]; then
  echo ""
  echo "Edge export:"
  if [ -f "${OUTPUT_DIR}/edge_export/model.safetensors" ]; then
    echo "✅ edge_export/model.safetensors"
  else
    echo "❌ edge_export/model.safetensors (missing)"
    ALL_GOOD=false
  fi
fi

echo ""
echo "Metrics:"
echo "==========================================="
if [ -f "${OUTPUT_DIR}/eval_metrics.json" ]; then
  cat "${OUTPUT_DIR}/eval_metrics.json"
fi

echo ""
echo "Cleaning up temporary files..."
rm -rf "${OUTPUT_DIR}"

echo ""
if [ "${ALL_GOOD}" = true ]; then
  echo "✅ All tests passed! Pipeline is working correctly."
  echo ""
  echo "You can now run full training with:"
  echo "  ${PYTHON_BIN} ml/training/train_cls.py --config ml/training/config/model.yaml"
else
  echo "❌ Some tests failed. Please check the errors above."
  exit 1
fi
