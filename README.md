# PyTorch Demo

A collection of simplified PyTorch programs demonstrating core deep learning concepts with minimal code. These examples are inspired by the [pytorch-deep-learning](https://github.com/mrdbourke/pytorch-deep-learning) course and showcase the fundamental principles that power modern AI systems.

## About This Project

This project takes lessons from the PyTorch Deep Learning course and creates simplified Python programs that demonstrate core concepts with the minimum amount of code necessary. The examples cover:

- **Data generation and preprocessing**
- **Train/test data splitting**
- **Neural network architecture**
- **Training with backpropagation**
- **Model evaluation and testing**
- **Large Language Model (LLM) basics**

These programs are highly simplified versions of what powers large language models (LLMs) like ChatGPT. While real LLMs use millions of parameters and train on vast datasets, these examples use simple data and minimal neural network layers to illustrate the same fundamental principles: generating data, training models through backpropagation, and optimizing until the model answers questions correctly.

## Project Files

### Setup Scripts

- **`setup.sh`** - One-time setup script that creates a virtual environment (`venv/`) and installs all required packages from `requirements.txt`. Checks for Python 3, creates the venv if needed, and verifies the installation. Safe to run multiple times.

- **`activate.sh`** - Helper script to activate the virtual environment. Simply run `source activate.sh` instead of typing `source venv/bin/activate`.

- **`sync_packages.sh`** - Syncs packages from `venv/` to `.venv/` so notebooks and scripts use the same packages. Useful when working with both Jupyter notebooks and Python scripts.

### Minimal LLM Examples

Introduction to creating Large Language Models from scratch:

- **`minimal_llm_0.py`** - Basic transformer architecture with character-level tokenization
- **`minimal_llm_1.py`** - Expanded transformer with training loop
- **`minimal_llm_2.py`** - Complete minimal LLM with text generation capabilities

These demonstrate the core transformer architecture (attention mechanisms, embeddings, language model head) in minimal code.

### PyTorch ML Examples

Simplified PyTorch programs based on [pytorch-deep-learning](https://github.com/mrdbourke/pytorch-deep-learning):

- **`simple_pytorch_example_00.py`** - Basic PyTorch fundamentals
- **`simple_pytorch_example_01.py`** - PyTorch workflow example
- **`simple_pytorch_example_02.py`** - Classification example with moons dataset
- **`simple_pytorch_example_03.py`** - Computer vision example using FashionMNIST (clothing classification with grainy 28x28 images)
- **`simple_pytorch_example_04.py`** - Custom datasets example with pizza/steak/sushi image classification
- **`simple_pytorch_example_05.py`** - Data augmentation and model training
- **`simple_pytorch_example_06.py`** - Additional PyTorch example

Each program includes detailed descriptions printed at the beginning explaining what it demonstrates.

### Full-Featured Example

- **`clothing-900-parameters-03-slow.py`** - Full version of `simple_pytorch_example_03.py`, based on `pytorch-deep-learning/03_pytorch_computer_vision.ipynb`. Classifies clothing using the FashionMNIST dataset (70,000 grainy 28x28 images of 10 clothing categories). This version trains for 900 epochs and requires significant computational resources - a Mac M4 Mini Pro or better is recommended. Runs very slowly on less powerful machines.

### Utility Modules

- **`imports.py`** - Centralized import module containing all imports required by the `simple_pytorch_example_*.py` programs. This includes PyTorch, torchvision, NumPy, Matplotlib, scikit-learn, PIL, and other dependencies. All example programs use `from imports import *` to ensure consistent imports across the project.

- **`my_utils.py`** - Utility functions used across the examples, including:
  - Training and testing loops (`train_loop`, `test_train_loop`)
  - DataLoader creation (`create_dataloaders`)
  - Accuracy calculation (`accuracy_fn`)
  - Model visualization and plotting utilities
  - Data download helpers
  - Device detection (CPU/CUDA/MPS)
  
  This module is imported by all example programs to avoid code duplication.

### Dependencies

- **`requirements.txt`** - Specifies exact versions of all Python packages needed. Includes:
  - **PyTorch ecosystem** (torch, torchvision, torchaudio) - Core deep learning framework
  - **Data science** (numpy, pandas, scikit-learn) - Numerical computing and ML utilities
  - **Visualization** (matplotlib, matplotlib-inline) - Plotting and graphs
  - **Image processing** (Pillow) - Image loading and manipulation
  - **Jupyter support** (ipykernel) - Notebook compatibility
  - **Utilities** (requests, tqdm, torchinfo) - Network requests, progress bars, model inspection

  Version numbers are pinned for reproducibility. All packages are installed automatically by `setup.sh`.

## Quick Start

### Initial Setup

Run the setup script to create a virtual environment and install all dependencies:

```bash
./setup.sh
```

This script will:
- Check for Python 3
- Create a virtual environment (`venv/`)
- Install all required packages from `requirements.txt`
- Verify the installation

Once setup completes, you'll see a message confirming you can run the Python programs.

### Running Programs

After setup, activate the virtual environment and run any of the example programs:

```bash
# Activate the virtual environment
source venv/bin/activate

# Run an example program
python simple_pytorch_example_00.py
python minimal_llm_0.py
# etc.
```

### Activating the Virtual Environment (For Future Sessions)

You only need to run `./setup.sh` once (or when you need to reinstall packages). For future terminal sessions, just activate the virtual environment:

```bash
source venv/bin/activate
```

Or use the activation script:

```bash
source activate.sh
```

## Project Structure

```
pytorch-demo/
├── setup.sh                           # One-time setup script (creates venv, installs packages)
├── activate.sh                        # Helper script to activate virtual environment
├── sync_packages.sh                   # Sync packages between venv and .venv
├── requirements.txt                   # Python package dependencies with pinned versions
├── imports.py                         # Centralized imports module for all example programs
├── my_utils.py                        # Utility functions for training and data handling
├── minimal_llm_*.py                   # Introduction to LLM creation (3 examples)
├── simple_pytorch_example_0*.py       # PyTorch ML examples based on pytorch-deep-learning
├── clothing-900-parameters-03-slow.py # Full clothing classification (requires M4 Mini Pro+)
└── data/                              # Training data (FashionMNIST, pizza/steak/sushi images)
```

## Example Programs Summary

See the "Project Files" section above for detailed descriptions. Quick reference:

**Minimal LLM Examples:**
- `minimal_llm_0.py`, `minimal_llm_1.py`, `minimal_llm_2.py` - Progressive introduction to transformer-based language models

**PyTorch ML Examples:**
- `simple_pytorch_example_00.py` through `simple_pytorch_example_06.py` - PyTorch deep learning fundamentals

**Full Example:**
- `clothing-900-parameters-03-slow.py` - Complete FashionMNIST clothing classification (computationally intensive)

## Dependencies

The project requires Python 3.8+ and the following key packages:

- **PyTorch** (torch, torchvision, torchaudio) - Deep learning framework
- **NumPy** - Numerical computing
- **Matplotlib** - Data visualization
- **Pandas** - Data manipulation
- **scikit-learn** - Machine learning utilities
- **Pillow** - Image processing
- **tqdm** - Progress bars

See `requirements.txt` for the complete list with version specifications.

## Verifying Installation

After running `./setup.sh`, verify everything is working:

```bash
source venv/bin/activate
python -c "import torch; print('✅ PyTorch:', torch.__version__)"
python -c "import matplotlib; print('✅ Matplotlib works!')"
python -c "import numpy; print('✅ NumPy works!')"
```

## Troubleshooting

### Virtual Environment Issues

If you encounter issues with the virtual environment:

1. **Broken or incomplete venv**: Run `./setup.sh` again. It will detect and recreate a broken virtual environment.

2. **Packages not found**: Make sure the virtual environment is activated:
   ```bash
   source venv/bin/activate
   ```
   You should see `(venv)` in your terminal prompt.

### Reinstalling Dependencies

To reinstall all packages:

```bash
source venv/bin/activate
pip install -r requirements.txt --upgrade
```

Or simply run `./setup.sh` again (it's safe to run multiple times).

### Multiprocessing Issues on macOS

If you see errors about multiprocessing when running programs with `num_workers > 0` in DataLoader:

- Make sure your Python scripts use `if __name__ == '__main__':` to protect the main execution code
- Alternatively, set `num_workers=0` in DataLoader creation (slower but avoids multiprocessing issues)

### Package Import Errors

If Python can't find packages even after activation:

1. Verify the virtual environment is activated: `which python` should point to `venv/bin/python`
2. Check package installation: `pip list | grep torch`
3. Reinstall packages: `pip install -r requirements.txt`

## Notes

- The virtual environment (`venv/`) is stored in the project directory and persists across restarts
- Always activate the virtual environment before running scripts or working with notebooks
- The setup script is idempotent - you can run it multiple times safely
- On macOS, PyTorch will use MPS (Metal Performance Shaders) if available for GPU acceleration

## Additional Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [PyTorch Deep Learning Course](https://github.com/mrdbourke/pytorch-deep-learning)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
