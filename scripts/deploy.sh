#!/usr/bin/env bash
#
# Deploy injection detector to Fastly Compute
# Usage: ./scripts/deploy.sh [--build-only]
#

set -euo pipefail

BUILD_ONLY="${1:-}"

echo "==========================================="
echo "Fastly Compute Deployment Pipeline"
echo "==========================================="
echo ""

# Check for Fastly CLI
if ! command -v fastly &> /dev/null; then
    echo "Error: Fastly CLI not found. Install from: https://docs.fastly.com/cli/"
    exit 1
fi

# Check for Rust
if ! command -v cargo &> /dev/null; then
    echo "Error: Rust not found. Install from: https://rustup.rs/"
    exit 1
fi

# Install wasm32 target if needed
echo "Checking Rust WebAssembly target..."
rustup target add wasm32-wasip1 2>/dev/null || true

# Check model size
if [ -f "service/assets/injection_1x128_int8.onnx" ]; then
    MODEL_SIZE=$(du -h service/assets/injection_1x128_int8.onnx | cut -f1)
    echo "Model size: $MODEL_SIZE"

    SIZE_BYTES=$(du -b service/assets/injection_1x128_int8.onnx | cut -f1)
    if [ $SIZE_BYTES -gt $((128 * 1024 * 1024)) ]; then
        echo "WARNING: Model exceeds 128MB limit. Build may fail."
    fi
else
    echo "Note: No model found at service/assets/injection_1x128_int8.onnx."
fi

# Build the Fastly package using fastly.toml (includes all correct features)
echo "Building Fastly Compute package..."
cd service

echo "Running fastly compute build (reads fastly.toml for correct build flags)..."
fastly compute build

# Report binary size (fastly compute build puts package at pkg/injection-detector.tar.gz)
if [ -f "pkg/injection-detector.tar.gz" ]; then
    PKG_SIZE=$(du -h pkg/injection-detector.tar.gz | cut -f1)
    echo "Package size: $PKG_SIZE"
fi

if [ "$BUILD_ONLY" == "--build-only" ]; then
    echo ""
    echo "Build complete! Package at: service/pkg/injection-detector.tar.gz"
    echo "To deploy manually: cd service && fastly compute deploy"
    exit 0
fi

# Deploy to Fastly
echo ""
echo "Deploying to Fastly Compute..."
fastly compute deploy --service-id "${FASTLY_SERVICE_ID}" --accept-defaults

cd ..

echo ""
echo "Deployment complete!"
echo "Your injection detector is now live on Fastly Compute."
echo ""
echo "API Endpoints:"
echo "  GET  /health     - Health check"
echo "  POST /classify   - Classify text for injection"
echo ""
echo "Example usage:"
echo '  curl -X POST https://your-service.edgecompute.app/classify \'
echo '    -H "Content-Type: application/json" \'
echo '    -d '"'"'{"text": "Your text to classify"}'"'"
