#!/usr/bin/env python

from imports import *

DATA_PATH = Path("data/")
IMAGE_PATH = DATA_PATH / "pizza_steak_sushi"
TRAIN_DIR = IMAGE_PATH / "train"
TEST_DIR = IMAGE_PATH / "test"

def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}.")
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

class ImageFolderCustom(Dataset):
    
    def __init__(self, targ_dir: str, transform=None) -> None:
        
        self.paths = list(pathlib.Path(targ_dir).glob("*/*.jpg")) 
        self.transform = transform
        self.classes, self.class_to_idx = find_classes(targ_dir)

    def load_image(self, index: int) -> Image.Image:
        "Opens an image via a path and returns it."
        image_path = self.paths[index]
        return Image.open(image_path) 
    
    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        "Returns one sample of data, data and label (X, y)."
        img = self.load_image(index)
        class_name  = self.paths[index].parent.name 
        class_idx = self.class_to_idx[class_name]

        if self.transform:
            return self.transform(img), class_idx 
        else:
            return img, class_idx 

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

    print("")
    my_utils.pretty_print("""
    This program summarizes the notebooks from 04_pytorch_custom_datasets.ipynb onwards.
It reads a collection of images of pizza, sushi, and steak, and tries to classify them.
The program uses a subset of the full PyTorch food dataset.
To use the full dataset, download it by uncommenting this line: my_utils.get_pizza_steak_sushi_data().
    """)
    # my_utils.get_pizza_steak_sushi_data()
    print("")

    data_transform = transforms.Compose([ 
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    train_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    train_dataloader, test_dataloader, class_names = my_utils.create_dataloaders(train_dir=TRAIN_DIR, 
                                                                                 test_dir=TEST_DIR, 
                                                                                 transform=data_transform,
                                                                                 batch_size=32, 
                                                                                 num_workers=4)

    model = TinyVGG(input_shape=3, hidden_units=10, output_shape=len(class_names))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    my_utils.test_train_loop( model=model, 
                              train_dataloader=train_dataloader, 
                              test_dataloader=test_dataloader, 
                              loss_fn=loss_fn, 
                              optimizer=optimizer, 
                              accuracy_fn=my_utils.accuracy_fn, 
                              epochs=10 )

    print("")

    train_data_custom = ImageFolderCustom(targ_dir=TRAIN_DIR, 
                                        transform=train_transforms)
    test_data_custom = ImageFolderCustom(targ_dir=TEST_DIR, 
                                        transform=test_transforms)

    device = my_utils.get_device()

    for dirpath, dirnames, filenames in os.walk(DATA_PATH):
        if re.match( r".+/pizza_steak_sushi/.+/.+", dirpath):
            print( f"{len(filenames)} images in '{dirpath}'.")

    image_path_list = list(IMAGE_PATH.glob("*/*/*.jpg"))

    data_transform = transforms.Compose([
        transforms.Resize(size=(64, 64)),
        transforms.RandomHorizontalFlip(p=0.5), 
        transforms.ToTensor()
    ])

    print("\nA random image from the food dataset...")
    random_image_paths = random.sample(image_path_list, k=3)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(1, 1)
            ax.imshow(f) 
            ax.set_title(f"\nSize: {f.size}")
            ax.axis("off")
            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)
            plt.show()
            break;

if __name__ == '__main__':
    main()


