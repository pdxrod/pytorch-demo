#!/usr/bin/env python

from imports import *

DATA_PATH = Path("data/")
IMAGE_PATH = DATA_PATH / "pizza_steak_sushi"
TRAIN_DIR = IMAGE_PATH / "train"
TEST_DIR = IMAGE_PATH / "test"

def intro():
    print("")
    my_utils.pretty_print("""
This program is intended to summarize the notebook pytorch-deep-learning/ 
04_pytorch_custom_datasets.ipynb thru 07_pytorch_experiment_tracking.ipynb.
It reads a collection of images of pizza, sushi, and steak, and tries to classify them.
It recreates class TinyVGG, a small-scale version of a convolutional neural network,
then saves the model to a file.
See https://poloclub.github.io/cnn-explainer/.
The program uses a subset of the full PyTorch food dataset.
To use the full dataset, download it by uncommenting this line: my_utils.get_pizza_steak_sushi_data().
    """)
    # my_utils.get_pizza_steak_sushi_data()
    print("")

def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}.")
#    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes #, class_to_idx

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
                    kernel_size=3, 
                    stride=1, 
                    padding=0), 
          nn.ReLU(),
          nn.Conv2d(in_channels=hidden_units, 
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=0),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2,
                        stride=2)
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
          nn.Linear(in_features=hidden_units*13*13,
                    out_features=output_shape)
      )
    
  def forward(self, x: torch.Tensor):
      x = self.conv_block_1(x)
      x = self.conv_block_2(x)
      x = self.classifier(x)
      return x
from pathlib import Path

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True,
                        exist_ok=True)
  
  model_save_path = target_dir_path / model_name
  print(f"Saving model to: {model_save_path}")
  torch.save(obj=model.state_dict(), f=model_save_path)

def display_random_images(dataset: torch.utils.data.Dataset,
                          classes: List[str] = None,
                          n: int = 5,
                          display_shape: bool = True,
                          seed: int = None):
  if n > 10:
    n = 10
    display_shape = False
    print(f"For display, purposes, n shouldn't be larger than 10, setting to 10 and removing shape display.")

  if seed:
    random.seed(seed)

  k = n
  l = len(dataset)
  if( k > l ):
    k = l

  random_samples_idx = random.sample(range( l ), k=k)

  # Calculate grid dimensions for better layout
  cols = min(5, k)  # Max 5 columns
  rows = (k + cols - 1) // cols  # Calculate rows needed
  
  fig, axes = plt.subplots(rows, cols, figsize=(15, 3*rows))
  
  # Ensure axes is always a 2D array for consistent indexing
  if rows == 1 and cols == 1:
    axes = np.array([[axes]])
  elif rows == 1:
    axes = axes.reshape(1, -1)
  elif cols == 1:
    axes = axes.reshape(-1, 1)
  else:
    axes = axes.reshape(rows, cols)

  for i, targ_sample in enumerate(random_samples_idx):
    targ_image, targ_label = dataset[targ_sample][0], dataset[targ_sample][1]

    targ_image_adjust = targ_image.permute(1, 2, 0) 

    row = i // cols
    col = i % cols
    ax = axes[row, col]
    
    ax.imshow(targ_image_adjust)
    ax.axis("off")
    if classes:
      title = f"Class: {classes[targ_label]}"
      if display_shape:
        title = title + f"\nshape: {targ_image_adjust.shape}"
      ax.set_title(title)
  
  # Hide any unused subplots
  for i in range(k, rows * cols):
    row = i // cols
    col = i % cols
    axes[row, col].axis("off")
  
  plt.tight_layout()
  plt.show()

def main():
    intro()

    data_transform = transforms.Compose([ transforms.Resize((64, 64)), transforms.ToTensor(), ])
    train_dataloader, test_dataloader, class_names = my_utils.create_dataloaders( train_dir=TRAIN_DIR, 
                                                                                  test_dir=TEST_DIR, 
                                                                                  transform=data_transform,
                                                                                  batch_size=32, 
                                                                                  num_workers=4 )

    model = TinyVGG(input_shape=3, hidden_units=10, output_shape=len(class_names))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    accuracy_fn = my_utils.accuracy_fn

    for dirpath, dirnames, filenames in os.walk(DATA_PATH):
        if re.match( r".+/pizza_steak_sushi/.+/.+", dirpath):
            print( f"{len(filenames)} images in '{dirpath}'.")

    my_utils.test_train_loop( model=model, 
                              train_dataloader=train_dataloader, 
                              test_dataloader=test_dataloader, 
                              loss_fn=loss_fn, 
                              optimizer=optimizer, 
                              accuracy_fn=accuracy_fn, 
                              epochs=10 )
    print("")
    my_utils.wait_for_user_input("Save the model to a file...")
    save_model(model=model, target_dir="models", model_name="simple_example_04.pth")
    print("")

    my_utils.wait_for_user_input("Display random images from the food dataset")
    classes = find_classes(directory=TRAIN_DIR)
    display_random_images(dataset =train_dataloader.dataset,
                          classes =classes,
                          n = 5,
                          display_shape = True,
                          seed = None)

if __name__ == '__main__':
    main()
