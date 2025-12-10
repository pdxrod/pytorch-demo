#!/bin/bash
# Install all packages from requirements.txt in both .venv and venv

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check which venv directories exist
HAS_DOTVENV=false
HAS_VENV=false

if [ -d ".venv" ]; then
    HAS_DOTVENV=true
    echo "Installing packages in .venv (for notebook)..."
    .venv/bin/pip install -r requirements.txt
    echo "✅ .venv packages installed"
fi

if [ -d "venv" ]; then
    HAS_VENV=true
    echo "Installing packages in venv (for terminal scripts)..."
    venv/bin/pip install -r requirements.txt
    echo "✅ venv packages installed"
fi

echo ""
echo "Verifying installations..."

if [ "$HAS_DOTVENV" = true ]; then
    .venv/bin/python -c "import torch; print('✅ torch works in .venv (notebook)')" 2>&1 || echo "❌ torch missing in .venv"
    .venv/bin/python -c "from tqdm.auto import tqdm; print('✅ tqdm works in .venv')" 2>&1 || echo "❌ tqdm missing in .venv"
fi

if [ "$HAS_VENV" = true ]; then
    venv/bin/python -c "import torch; print('✅ torch works in venv (terminal)')" 2>&1 || echo "❌ torch missing in venv"
    venv/bin/python -c "from tqdm.auto import tqdm; print('✅ tqdm works in venv')" 2>&1 || echo "❌ tqdm missing in venv"
fi

echo ""
echo "Done! If the notebook still can't find packages, run this in a notebook cell:"
echo "  import sys; print(sys.executable)"
echo "Then install packages in that Python path."

