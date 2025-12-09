# pytorch-demo

An attempt to take the lessons in pytorch-deep-learning https://github.com/mrdbourke/pytorch-deep-learning, and create python programs which show the lessons with the miniumum amound of code. 

Real LLMs are a massively larger version of this code. The basic steps - generating data, splitting into train and test data, 'backpropagation', where the software changes numbers in neurons until the model answers a question correctly, are the same. A big LLM like ChatGPT does the same millions of times, with half of the internet and millions of books, papers, images and clips, until it is giving the right answer over 99% of the time to a vast range of questions it has never seen before - then it's considered trained. These programs are highly simplified version of this process - simple data, only three layers of neurons - but the principle is the same. 

##

# Project Setup Instructions

This project uses a Python virtual environment to ensure consistent package versions.

## Quick Start

### Option 1: Activate the virtual environment manually
```bash
source venv/bin/activate
```

### Option 2: Use the activation script
```bash
source activate.sh
```

### Option 3: Use the activation script (from anywhere)
```bash
cd /Users/rod/dev/ml/pytorch-demo
source activate.sh
```

## Running Notebooks

After activating the virtual environment, make sure your notebook is using the correct kernel:

1. Open your notebook in Cursor/Jupyter
2. Select the kernel: **"Python (pytorch-demo)"**
   - In Jupyter: Kernel → Change Kernel → Python (pytorch-demo)
   - In VS Code/Cursor: Click the kernel selector in the top right, choose "Python (pytorch-demo)"

## Verifying Installation

To verify everything is working:
```bash
source venv/bin/activate
python -c "import matplotlib; print('✅ matplotlib works!')"
python -c "import torch; print('✅ PyTorch works!')"
```

## Reinstalling Dependencies

If you need to reinstall packages:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

## Making it Persistent

The virtual environment is already in your project directory, so it will persist across restarts. Just make sure to:

1. Activate the virtual environment before running notebooks or scripts
2. Select the correct kernel in your notebook interface
3. The environment will persist because it's stored in the `venv/` directory

## Troubleshooting

If matplotlib or other packages can't be found:
1. Make sure the virtual environment is activated: `source venv/bin/activate`
2. Check you're using the right kernel in your notebook
3. Reinstall packages: `pip install -r requirements.txt`

