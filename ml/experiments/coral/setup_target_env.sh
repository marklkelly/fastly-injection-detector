#!/bin/bash

# Setup script for Edge TPU deployment environment
# This script sets up Python, virtual environment, and all dependencies for running
# the Edge TPU FFN models with inference harness

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Edge TPU Deployment Environment Setup${NC}"
echo -e "${BLUE}========================================${NC}"

# Check if running as root (needed for apt installs)
if [ "$EUID" -ne 0 ] && [ "$1" != "--no-root" ]; then
    echo -e "${YELLOW}Note: This script needs sudo for system packages.${NC}"
    echo "Re-running with sudo..."
    sudo "$0" "$@"
    exit $?
fi

# Function to check command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python version
check_python_version() {
    local python_cmd=$1
    if command_exists $python_cmd; then
        version=$($python_cmd -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        major=$(echo $version | cut -d. -f1)
        minor=$(echo $version | cut -d. -f2)

        if [ "$major" -eq 3 ] && [ "$minor" -ge 8 ] && [ "$minor" -le 11 ]; then
            echo $python_cmd
            return 0
        fi
    fi
    return 1
}

# 1. Check system and architecture
echo -e "\n${GREEN}1. System Information${NC}"
echo "OS: $(uname -s)"
echo "Architecture: $(uname -m)"
echo "Hostname: $(hostname)"

# 2. Check/Install Python
echo -e "\n${GREEN}2. Python Setup${NC}"

# Find suitable Python version (3.8-3.11 for Edge TPU compatibility)
PYTHON_CMD=""
for py in python3.9 python3.10 python3.8 python3.11 python3; do
    if check_python_version $py; then
        PYTHON_CMD=$py
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo -e "${YELLOW}No suitable Python found (need 3.8-3.11). Installing Python 3.9...${NC}"

    # Detect package manager and install Python
    if command_exists apt-get; then
        apt-get update
        apt-get install -y python3.9 python3.9-venv python3.9-dev python3-pip
        PYTHON_CMD="python3.9"
    elif command_exists yum; then
        yum install -y python39 python39-devel
        PYTHON_CMD="python3.9"
    else
        echo -e "${RED}Could not install Python. Please install Python 3.8-3.11 manually.${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}✓ Using Python: $PYTHON_CMD ($($PYTHON_CMD --version))${NC}"

# 3. Install system dependencies
echo -e "\n${GREEN}3. System Dependencies${NC}"

if command_exists apt-get; then
    echo "Installing Edge TPU runtime and dependencies..."

    # Add Coral repository if not already added
    if ! grep -q "coral-edgetpu-stable" /etc/apt/sources.list.d/coral-edgetpu.list 2>/dev/null; then
        echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list
        curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
    fi

    apt-get update

    # Install Edge TPU runtime (standard frequency)
    apt-get install -y libedgetpu1-std

    # Option for max frequency (uncomment if desired)
    # apt-get install -y libedgetpu1-max

    # Install other dependencies
    apt-get install -y libusb-1.0-0 libc++1 libc++abi1 libgcc1

    echo -e "${GREEN}✓ System dependencies installed${NC}"
else
    echo -e "${YELLOW}⚠ Non-Debian system detected. Please install Edge TPU runtime manually.${NC}"
    echo "Visit: https://coral.ai/docs/accelerator/get-started/#runtime"
fi

# 4. Create virtual environment
echo -e "\n${GREEN}4. Virtual Environment Setup${NC}"

VENV_DIR="venv_coral"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv $VENV_DIR
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
source $VENV_DIR/bin/activate

echo -e "${GREEN}✓ Virtual environment activated${NC}"

# 5. Upgrade pip
echo -e "\n${GREEN}5. Upgrading pip${NC}"
pip install --upgrade pip wheel setuptools

# 6. Install Python packages
echo -e "\n${GREEN}6. Installing Python packages${NC}"

# Detect architecture and OS for tflite-runtime
ARCH=$(uname -m)
OS=$(uname -s)

echo "Detected: OS=$OS, Architecture=$ARCH"

# Install tflite-runtime based on platform
if [ "$OS" = "Linux" ]; then
    # Get Python version for wheel selection
    PY_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')

    if [ "$ARCH" = "x86_64" ]; then
        echo "Installing tflite-runtime for Linux x86_64..."
        # Try direct wheel URL for common Python versions
        if [ "$PY_VERSION" = "39" ]; then
            pip install https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp39-cp39-linux_x86_64.whl || \
            pip install tflite-runtime
        elif [ "$PY_VERSION" = "38" ]; then
            pip install https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp38-cp38-linux_x86_64.whl || \
            pip install tflite-runtime
        elif [ "$PY_VERSION" = "310" ]; then
            pip install https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp310-cp310-linux_x86_64.whl || \
            pip install tflite-runtime
        else
            pip install tflite-runtime
        fi
    elif [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
        echo "Installing tflite-runtime for ARM64..."
        pip install https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp39-cp39-linux_aarch64.whl || \
        pip install tflite-runtime
    else
        echo -e "${YELLOW}Attempting standard tflite-runtime install...${NC}"
        pip install tflite-runtime
    fi
elif [ "$OS" = "Darwin" ]; then
    echo -e "${YELLOW}macOS detected. Installing tensorflow-lite instead of tflite-runtime...${NC}"
    # On macOS, use tensorflow instead
    pip install tensorflow
else
    echo -e "${YELLOW}Unknown OS. Attempting standard install...${NC}"
    pip install tflite-runtime
fi || {
    echo -e "${RED}Failed to install tflite-runtime.${NC}"
    echo -e "${YELLOW}You may need to install it manually or use tensorflow.${NC}"
}

# Install other core packages
echo "Installing core packages..."
pip install "numpy<2.0" || echo "NumPy install failed"

# Create requirements file for optional packages
cat > requirements_optional.txt << EOF
# Optional but recommended
Pillow
scipy
scikit-learn

# For tokenization (optional - will fall back to simple tokenization if not available)
tokenizers
transformers

# For HTTP server (optional)
flask
EOF

echo "Installing optional packages (failures are OK)..."
pip install -r requirements_optional.txt 2>/dev/null || {
    echo -e "${YELLOW}Some optional packages failed. Core packages should be sufficient.${NC}"
}

# 7. Test Edge TPU detection
echo -e "\n${GREEN}7. Testing Edge TPU Detection${NC}"

python << EOF
import sys
try:
    import tflite_runtime.interpreter as tflite
    print("✓ TFLite Runtime imported successfully")

    # Try to detect Edge TPU
    try:
        delegate = tflite.load_delegate('libedgetpu.so.1')
        print("✓ Edge TPU delegate loaded successfully")

        # Check if USB Coral is connected
        import subprocess
        result = subprocess.run(['lsusb'], capture_output=True, text=True)
        if '1a6e:089a' in result.stdout or 'Global Unichip' in result.stdout:
            print("✓ USB Coral detected")
        else:
            print("⚠ No USB Coral detected (may be using M.2 or not connected)")

    except Exception as e:
        print(f"⚠ Edge TPU not available: {e}")
        print("  If you have a Coral device, make sure it's connected")

except ImportError as e:
    print(f"✗ Failed to import tflite_runtime: {e}")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Python environment validated${NC}"
else
    echo -e "${RED}✗ Python environment validation failed${NC}"
    exit 1
fi

# 8. Extract deployment bundle if it exists
echo -e "\n${GREEN}8. Deployment Bundle${NC}"

if [ -f "edge_tpu_deployment_bundle_v2.tar.gz" ]; then
    echo "Extracting deployment bundle..."
    tar xzf edge_tpu_deployment_bundle_v2.tar.gz
    echo -e "${GREEN}✓ Bundle extracted${NC}"
elif [ -f "edge_tpu_deployment_bundle.tar.gz" ]; then
    echo "Extracting deployment bundle..."
    tar xzf edge_tpu_deployment_bundle.tar.gz
    echo -e "${GREEN}✓ Bundle extracted${NC}"
else
    echo -e "${YELLOW}⚠ No deployment bundle found. Please copy the bundle to this directory.${NC}"
fi

# 9. Create test script
echo -e "\n${GREEN}9. Creating test script${NC}"

cat > test_deployment.py << 'EOF'
#!/usr/bin/env python3
"""Quick test of Edge TPU deployment."""

import sys
import numpy as np

print("Testing Edge TPU Deployment")
print("-" * 40)

# Test imports
try:
    import tflite_runtime.interpreter as tflite
    print("✓ TFLite runtime imported")
except ImportError:
    print("✗ TFLite runtime not available")
    sys.exit(1)

# Test Edge TPU
try:
    delegate = tflite.load_delegate('libedgetpu.so.1')
    print("✓ Edge TPU delegate loaded")
    tpu_available = True
except:
    print("⚠ Edge TPU not available (will use CPU)")
    tpu_available = False

# Test model loading if available
import os
if os.path.exists("models_delta/ffn0_delta_int8_edgetpu.tflite"):
    try:
        if tpu_available:
            interpreter = tflite.Interpreter(
                model_path="models_delta/ffn0_delta_int8_edgetpu.tflite",
                experimental_delegates=[delegate]
            )
        else:
            interpreter = tflite.Interpreter(
                model_path="models_delta/ffn0_delta_int8_edgetpu.tflite"
            )
        interpreter.allocate_tensors()
        print("✓ Model loaded successfully")

        # Test inference
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]

        test_input = np.random.randint(-128, 127, size=input_details['shape']).astype(np.int8)
        interpreter.set_tensor(input_details['index'], test_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details['index'])

        print(f"✓ Test inference completed")
        print(f"  Input shape: {input_details['shape']}")
        print(f"  Output shape: {output_details['shape']}")

    except Exception as e:
        print(f"✗ Model test failed: {e}")
else:
    print("⚠ Model files not found")

print("-" * 40)
print("Setup complete! You can now run:")
print("  python3 inference_harness.py --test")
print("  python3 inference_harness.py --server")
EOF

chmod +x test_deployment.py

# 10. Final summary
echo -e "\n${BLUE}========================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${BLUE}========================================${NC}"

echo -e "\n${GREEN}Environment:${NC}"
echo "  Python: $PYTHON_CMD"
echo "  Virtual env: $VENV_DIR"
echo "  Activation: source $VENV_DIR/bin/activate"

echo -e "\n${GREEN}Next steps:${NC}"
echo "1. Test the deployment:"
echo "   python test_deployment.py"
echo ""
echo "2. Run inference harness:"
echo "   python inference_harness.py --test"
echo ""
echo "3. Start HTTP server:"
echo "   python inference_harness.py --server --port 8080"
echo ""
echo "4. For max TPU frequency (optional):"
echo "   sudo apt-get install libedgetpu1-max"

if [ "$EUID" -eq 0 ]; then
    echo -e "\n${YELLOW}Note: This script was run as root. The virtual environment${NC}"
    echo -e "${YELLOW}may have root ownership. Consider chowning to your user.${NC}"
fi

echo -e "\n${GREEN}✓ All done!${NC}"