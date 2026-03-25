#!/bin/bash
# Setup script for Coral Edge TPU development environment
# PyCoral requires Python 3.6-3.9 (3.9 recommended)

set -e

echo "=========================================="
echo "Coral Edge TPU Environment Setup"
echo "=========================================="

# Check current Python version
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

echo "Current Python version: $PYTHON_VERSION"

# Check if Python version is compatible (3.6-3.9)
if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 6 ] && [ "$PYTHON_MINOR" -le 9 ]; then
    echo "✅ Python version is compatible with PyCoral"
    PYTHON_CMD="python3"
else
    echo "❌ Python $PYTHON_VERSION is not compatible with PyCoral"
    echo "   PyCoral requires Python 3.6-3.9"
    echo ""

    # Check for Python 3.9 specifically
    if command -v python3.9 &> /dev/null; then
        echo "✅ Found Python 3.9"
        PYTHON_CMD="python3.9"
    else
        echo "Please install Python 3.9 using one of these methods:"
        echo ""
        echo "Option 1: Using pyenv (recommended)"
        echo "  brew install pyenv"
        echo "  pyenv install 3.9.18"
        echo "  pyenv local 3.9.18"
        echo ""
        echo "Option 2: Using homebrew"
        echo "  brew install python@3.9"
        echo ""
        echo "Option 3: Using conda"
        echo "  conda create -n coral python=3.9"
        echo "  conda activate coral"
        echo ""
        exit 1
    fi
fi

# Create virtual environment with compatible Python
echo ""
echo "Creating virtual environment..."
if [ -d ".venv_coral" ]; then
    echo "Virtual environment already exists (.venv_coral)"
else
    $PYTHON_CMD -m venv .venv_coral
    echo "✅ Created .venv_coral"
fi

# Activate virtual environment
source .venv_coral/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
python -m pip install --upgrade pip --quiet

# Install PyCoral and dependencies
echo ""
echo "Installing PyCoral and dependencies..."
python -m pip install --extra-index-url https://google-coral.github.io/py-repo/ pycoral~=2.0

# Install other required packages
echo ""
echo "Installing additional packages..."
python -m pip install numpy pillow tflite-runtime

# Test installation
echo ""
echo "Testing PyCoral installation..."
python -c "
try:
    from pycoral.utils.edgetpu import list_edge_tpus, get_runtime_version
    print('✅ PyCoral imported successfully')

    # Try to get runtime version
    try:
        version = get_runtime_version()
        print(f'   Edge TPU Runtime version: {version}')
    except:
        print('   Could not get runtime version (runtime may not be installed)')

    # List TPUs
    tpus = list_edge_tpus()
    if tpus:
        print(f'✅ Found {len(tpus)} Edge TPU(s):')
        for i, tpu in enumerate(tpus):
            print(f'   TPU {i}: {tpu}')
    else:
        print('⚠️  No Edge TPUs detected')
        print('   - Check USB connection')
        print('   - Try unplugging and replugging the Coral')
        print('   - Ensure Edge TPU runtime is installed')

except ImportError as e:
    print(f'❌ Failed to import PyCoral: {e}')
except Exception as e:
    print(f'❌ Error: {e}')
"

echo ""
echo "=========================================="
echo "Setup complete!"
echo ""
echo "To activate the environment in the future:"
echo "  source .venv_coral/bin/activate"
echo ""
echo "To test Coral detection:"
echo "  python -c 'from pycoral.utils.edgetpu import list_edge_tpus; print(list_edge_tpus())'"
echo "=========================================="