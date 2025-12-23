#!/usr/bin/env python

from imports import *

DATA_PATH = Path("data/")
IMAGE_PATH = DATA_PATH / "pizza_steak_sushi"
TRAIN_DIR = IMAGE_PATH / "train"
TEST_DIR = IMAGE_PATH / "test"

def intro():
    print("")
    my_utils.pretty_print("""
This program is intended to summarize the notebooks from pytorch-deep-learning/ 04_pytorch_custom_datasets.ipynb onwards.
It reads a collection of images of pizza, sushi, and steak, and tries to classify them.
It recreates class TinyVGG, a small-scale version of a convolutional neural network.
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
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

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

    image_path_list = list(IMAGE_PATH.glob("*/*/*.jpg"))
    random_image_paths = random.sample(image_path_list, k=3 )

    print("\nA random image from the food dataset...")
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(1, 1)
            ax.imshow(f) 
            ax.set_title(f"\nSize: {f.size}")
            ax.axis("off")
            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)
            plt.show()
            break

if __name__ == '__main__':
    main()
