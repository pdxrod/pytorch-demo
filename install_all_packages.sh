#!/bin/bash
# Install all packages from requirements.txt in both .venv and venv

echo "Installing packages in .venv (for notebook)..."
.venv/bin/pip install -r requirements.txt

echo ""
echo "Installing packages in venv (for terminal scripts)..."
venv/bin/pip install -r requirements.txt

echo ""
echo "✅ Done! Verifying tqdm..."
.venv/bin/python -c "from tqdm.auto import tqdm; print('✅ tqdm works in .venv (notebook)')"
venv/bin/python -c "from tqdm.auto import tqdm; print('✅ tqdm works in venv (terminal)')"

echo ""
echo "Both environments now have the same packages!"

