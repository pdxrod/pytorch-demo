#!/bin/bash
# Install matplotlib and required packages in pytorch-demo/.venv

echo "Installing packages in /Users/rod/dev/ml/pytorch-demo/.venv..."
/Users/rod/dev/ml/pytorch-demo/.venv/bin/python -m pip install matplotlib numpy torch torchvision scikit-learn pandas Pillow requests

echo ""
echo "Verifying installation..."
/Users/rod/dev/ml/pytorch-demo/.venv/bin/python -c "import matplotlib; print('✅ matplotlib version:', matplotlib.__version__)"
/Users/rod/dev/ml/pytorch-demo/.venv/bin/python -c "import torch; print('✅ torch version:', torch.__version__)"

