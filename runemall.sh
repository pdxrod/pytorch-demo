#!/bin/sh

set -e

# PYTORCH_MPS_DISABLE=1 OMP_NUM_THREADS=1 python minimal_llm_0.py
echo "minimal_llm_0.py"
python minimal_llm_0.py
echo "minimal_llm_1.py"
python minimal_llm_1.py
echo "simple_pytorch_example_1.py"
python simple_pytorch_example_1.py
echo "simple_pytorch_example_2.py"
python simple_pytorch_example_2.py
echo "simple_pytorch_example_3.py"
python simple_pytorch_example_3.py
echo "simple_pytorch_example_4.py"
python simple_pytorch_example_4.py
echo "simple_pytorch_example_5.py"
python simple_pytorch_example_5.py
echo "simple_pytorch_example_6.py"
python simple_pytorch_example_6.py

