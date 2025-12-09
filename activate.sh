#!/bin/bash
# Activate the virtual environment for this project
# Usage: source activate.sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/venv/bin/activate"
echo "✅ Virtual environment activated!"
echo "Python: $(which python)"
echo "Python version: $(python --version)"

