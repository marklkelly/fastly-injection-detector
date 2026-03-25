#!/bin/bash
# WARNING: This script requires a populated ./backups/last_deployment.txt file.
# No pipeline in this repo currently creates that file automatically.
# As a result, this script will exit with "No rollback information found!" on a fresh checkout.
#
# ALTERNATIVE: Use the Fastly CLI to roll back directly:
#   fastly service-version activate --version <previous-version-number> \
#     --service-id ${FASTLY_SERVICE_ID}
#
# To list available service versions:
#   fastly service-version list --service-id ${FASTLY_SERVICE_ID}
#
# For automated rollback to work, deploy.sh must be updated to write
# deployment metadata to ./backups/last_deployment.txt before deployment.
set -e

# Rollback script to restore previous deployment

echo "🔄 Fastly Injection Detector - Rollback"
echo "========================================"

# Configuration
BACKUP_DIR="./backups"
ROLLBACK_FILE="${BACKUP_DIR}/last_deployment.txt"
MODEL_PATH="service/assets/injection_1x128_int8.onnx"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check if rollback info exists
if [ ! -f "$ROLLBACK_FILE" ]; then
    print_error "No rollback information found at: $ROLLBACK_FILE"
    echo ""
    echo "This file is not automatically created by any pipeline."
    echo "To roll back using Fastly CLI directly:"
    echo "  fastly service-version list --service-id ${FASTLY_SERVICE_ID}"
    echo "  fastly service-version activate --version <N> --service-id ${FASTLY_SERVICE_ID}"
    exit 1
fi

# Load rollback info
source "$ROLLBACK_FILE"

echo ""
echo "📋 Last deployment info:"
echo "  - Timestamp: $timestamp"
echo "  - Backup path: $backup_path"
echo "  - Model type: $model_type"
echo "  - Deployed at: $deployed_at"
echo ""

# Confirm rollback
read -p "Rollback to this version? (y/N): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_warning "Rollback cancelled"
    exit 0
fi

# Check if backup exists
if [ ! -d "$backup_path" ]; then
    print_error "Backup directory not found: $backup_path"
    exit 1
fi

echo ""
echo "🔄 Starting rollback..."

# Restore model
if [ -f "$backup_path/injection_1x128_int8.onnx" ]; then
    cp "$backup_path/injection_1x128_int8.onnx" "$MODEL_PATH"
    print_status "Restored model"
else
    print_warning "No model backup found"
fi

# Determine features based on model type
if [ "$model_type" = "two_inputs" ]; then
    FEATURES="inference two_inputs_i64"
else
    FEATURES="inference"
fi

echo ""
echo "🔨 Rebuilding with $model_type configuration..."
cd service

cargo build --release --target wasm32-wasip1 --features "$FEATURES"

if [ $? -ne 0 ]; then
    print_error "Build failed during rollback!"
    cd ..
    exit 1
fi

cd ..
print_status "Build successful"

# Test locally
echo ""
echo "🧪 Testing rollback locally..."

cd service
fastly compute serve --skip-build &
SERVER_PID=$!
cd ..

sleep 3

# Test health
if curl -s http://127.0.0.1:7676/health | grep -q "OK"; then
    print_status "Health check passed"
else
    print_error "Health check failed after rollback"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

# Test classification
RESPONSE=$(curl -s -X POST http://127.0.0.1:7676/classify \
    -H "Content-Type: application/json" \
    -d '{"text": "Test after rollback"}')

if echo "$RESPONSE" | grep -q "label"; then
    print_status "Classification test passed"
else
    print_error "Classification test failed"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

kill $SERVER_PID 2>/dev/null

# Deploy rollback
echo ""
echo "🚀 Deploying rollback to Fastly..."
read -p "Deploy rollback to production? (y/N): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    cd service
    fastly compute deploy --service-id ${FASTLY_SERVICE_ID} --accept-defaults
    DEPLOY_STATUS=$?
    cd ..
    
    if [ $DEPLOY_STATUS -eq 0 ]; then
        print_status "Rollback deployed successfully!"
        
        # Update rollback file
        echo "" >> "$ROLLBACK_FILE"
        echo "rolled_back_at=$(date)" >> "$ROLLBACK_FILE"
        
        echo ""
        echo "✅ Rollback complete!"
        echo "  - Restored to: $timestamp deployment"
        echo "  - Model type: $model_type"
    else
        print_error "Rollback deployment failed!"
        exit 1
    fi
else
    print_warning "Rollback deployment cancelled"
    echo "Local rollback complete, but not deployed to production."
fi

echo ""
echo "✅ Done!"