#!/usr/bin/env python

from imports import *
import my_utils
from my_utils import test_train_loop

class TinyVGG(nn.Module):
  """Creates the TinyVGG architecture.

  Replicates the TinyVGG architecture from the CNN explainer website in PyTorch.
  See the original architecture here: https://poloclub.github.io/cnn-explainer/
  
  Args:
    input_shape: An integer indicating number of input channels.
    hidden_units: An integer indicating number of hidden units between layers.
    output_shape: An integer indicating number of output units.
  """
  def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
      super().__init__()
      self.conv_block_1 = nn.Sequential(
          nn.Conv2d(in_channels=input_shape, 
                    out_channels=hidden_units, 
                    kernel_size=3, # how big is the square that's going over the image?
                    stride=1, # default
                    padding=0), # options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number 
          nn.ReLU(),
          nn.Conv2d(in_channels=hidden_units, 
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=0),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2,
                        stride=2) # default stride value is same as kernel_size
      )
      self.conv_block_2 = nn.Sequential(
          nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
          nn.ReLU(),
          nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
          nn.ReLU(),
          nn.MaxPool2d(2)
      )
      self.classifier = nn.Sequential(
          nn.Flatten(),
          # Where did this in_features shape come from? 
          # It's because each layer of our network compresses and changes the shape of our inputs data.
          nn.Linear(in_features=hidden_units*13*13,
                    out_features=output_shape)
      )
    
  def forward(self, x: torch.Tensor):
      x = self.conv_block_1(x)
      x = self.conv_block_2(x)
      x = self.classifier(x)
      return x
      # return self.classifier(self.block_2(self.block_1(x))) # <- leverage the benefits of operator fusion

if __name__ == '__main__':
    DATA_PATH = Path("data/")
    IMAGE_PATH = DATA_PATH / "pizza_steak_sushi"
    TRAIN_DIR = IMAGE_PATH / "train"
    TEST_DIR = IMAGE_PATH / "test"

    data_transform = transforms.Compose([ 
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    train_dataloader, test_dataloader, class_names = my_utils.create_dataloaders(train_dir=TRAIN_DIR, 
                                                                                 test_dir=TEST_DIR, 
                                                                                 transform=data_transform,
                                                                                 batch_size=32, 
                                                                                 num_workers=4)

    model = TinyVGG(input_shape=3, hidden_units=10, output_shape=len(class_names))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    accuracy_fn = my_utils.accuracy_fn

    test_train_loop(model=model, 
                    train_dataloader=train_dataloader, 
                    test_dataloader=test_dataloader, 
                    loss_fn=loss_fn, 
                    optimizer=optimizer, 
                    accuracy_fn=accuracy_fn, 
                    epochs=10)





