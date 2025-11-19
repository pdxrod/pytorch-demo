import torch

def wait_for_user_input(msg=None):
    print("")
    if msg is not None:
        print(msg)
    input( f"Press Enter to continue..." )  

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps" 
    else:
        return "cpu" 

def sigmoid(x):
  return 1 / (1 + torch.exp(-x))

def ReLU(x):
  return torch.maximum(torch.tensor(0), x) 
