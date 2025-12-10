#!/bin/bash
# This script will install packages in whatever Python the notebook is using
# First, run this in a notebook cell to find the Python path:
#   import sys; print(sys.executable)

echo "To fix the notebook packages:"
echo ""
echo "1. Run this in a notebook cell:"
echo "   import sys; print(sys.executable)"
echo ""
echo "2. Then run this command with the Python path from step 1:"
echo "   <python_path> -m pip install -r requirements.txt"
echo ""
echo "Or if you tell me the Python path, I can run it for you!"

