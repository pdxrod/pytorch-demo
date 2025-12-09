#!/bin/bash
# Sync packages from venv to .venv (so notebook and scripts use same packages)

echo "Syncing packages from venv to .venv..."

# Get all packages from venv
venv/bin/pip freeze > /tmp/venv_requirements.txt

# Install them in .venv
echo "Installing packages in .venv..."
.venv/bin/pip install -r /tmp/venv_requirements.txt

echo ""
echo "✅ Done! Both environments should now have the same packages."
echo ""
echo "Verifying tqdm..."
.venv/bin/python -c "from tqdm.auto import tqdm; print('✅ tqdm works in .venv')"
venv/bin/python -c "from tqdm.auto import tqdm; print('✅ tqdm works in venv')"

