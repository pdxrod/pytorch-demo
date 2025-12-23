# PyTorch Demo

A collection of simplified PyTorch programs demonstrating core deep learning concepts with minimal code. These examples are inspired by the [pytorch-deep-learning](https://github.com/mrdbourke/pytorch-deep-learning) course and showcase the fundamental principles that power modern AI systems.

## About This Project

This project takes lessons from the PyTorch Deep Learning course and creates simplified Python programs that demonstrate core concepts with the minimum amount of code necessary. The examples cover:

- **Data generation and preprocessing**
- **Train/test data splitting**
- **Neural network architecture**
- **Training with backpropagation**
- **Model evaluation and testing**

These programs are highly simplified versions of what powers large language models (LLMs) like ChatGPT. While real LLMs use millions of parameters and train on vast datasets, these examples use simple data and minimal neural network layers to illustrate the same fundamental principles: generating data, training models through backpropagation, and optimizing until the model answers questions correctly.

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

# Run the programs
python simple_example_00.py
python simple_example_01.py
python simple_example_02.py
python simple_example_03.py
python simple_example_04.py

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
├── setup.sh              # One-time setup script (creates venv, installs packages)
├── activate.sh           # Helper script to activate virtual environment
├── requirements.txt      # Python package dependencies
├── imports.py            # Common imports module
├── my_utils.py           # Utility functions for training and data handling
├── simple_example_*.py   # Example programs demonstrating PyTorch concepts
├── clothing-900-parameters-03.py  # Clothing classification with 900 parameters
└── data/                 # Training data (FashionMNIST, pizza/steak/sushi images)
```

## Example Programs

- **simple_example_00.py**  
- **simple_example_01.py** 
- **simple_example_02.py** 
- **simple_example_03.py** 
- **simple_example_04.py** 
- **clothing-900-parameters-03.py** - Clothing classification model 

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
