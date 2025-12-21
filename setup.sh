#!/bin/bash
# Setup script for pytorch-demo project
# This script creates a virtual environment, installs dependencies, and prepares the environment
# Usage: ./setup.sh

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/venv"
REQUIREMENTS_FILE="$SCRIPT_DIR/requirements.txt"

echo "🔧 Setting up pytorch-demo environment..."
echo ""

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 is not installed. Please install Python 3 first."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Check if activate script exists (handles both regular directories and symlinks)
ACTIVATE_SCRIPT="$VENV_DIR/bin/activate"
if [ ! -f "$ACTIVATE_SCRIPT" ]; then
    if [ -e "$VENV_DIR" ]; then
        echo ""
        echo "⚠️  Virtual environment exists but appears incomplete or broken"
        echo "📦 Recreating virtual environment..."
        rm -rf "$VENV_DIR"
    else
        echo ""
        echo "📦 Creating virtual environment..."
    fi
    python3 -m venv "$VENV_DIR"
    echo "✅ Virtual environment created at $VENV_DIR"
    ACTIVATE_SCRIPT="$VENV_DIR/bin/activate"
else
    echo ""
    echo "✅ Virtual environment found"
fi

# Activate virtual environment
echo ""
echo "🔄 Activating virtual environment..."
source "$ACTIVATE_SCRIPT"

# Upgrade pip
echo ""
echo "⬆️  Upgrading pip..."
pip install --upgrade pip --quiet

# Install packages from requirements.txt
if [ -f "$REQUIREMENTS_FILE" ]; then
    echo ""
    echo "📥 Installing packages from requirements.txt..."
    pip install -r "$REQUIREMENTS_FILE"
    echo "✅ Packages installed"
else
    echo ""
    echo "⚠️  Warning: requirements.txt not found at $REQUIREMENTS_FILE"
fi

# Verify critical packages
echo ""
echo "🔍 Verifying installation..."

if python -c "import torch" 2>/dev/null; then
    echo "✅ torch is installed"
else
    echo "❌ torch is missing"
fi

if python -c "import torchvision" 2>/dev/null; then
    echo "✅ torchvision is installed"
else
    echo "❌ torchvision is missing"
fi

if python -c "from tqdm.auto import tqdm" 2>/dev/null; then
    echo "✅ tqdm is installed"
else
    echo "❌ tqdm is missing"
fi

if python -c "import matplotlib" 2>/dev/null; then
    echo "✅ matplotlib is installed"
else
    echo "❌ matplotlib is missing"
fi

if python -c "import numpy" 2>/dev/null; then
    echo "✅ numpy is installed"
else
    echo "❌ numpy is missing"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Setup complete! You can now run the Python programs."
echo ""
echo "To activate the virtual environment in the future, run:"
echo "   source venv/bin/activate"
echo ""
echo "Or use the activate script:"
echo "   source activate.sh"
echo ""
echo "Python location: $(which python)"
echo "Python version: $(python --version)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

