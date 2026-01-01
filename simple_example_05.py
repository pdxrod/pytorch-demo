#!/usr/bin/env python

from imports import *

DATA_PATH = Path("data/")
IMAGE_PATH = DATA_PATH / "pizza_steak_sushi"
TRAIN_DIR = IMAGE_PATH / "train"
TEST_DIR = IMAGE_PATH / "test"
BATCH_SIZE = 32
RANDOM_SEED = 42
NUM_WORKERS = os.cpu_count()

def intro():
    print("")
    my_utils.pretty_print("""
This program is intended to summarize the notebooks pytorch-deep-learning/ 
04_pytorch_custom_datasets.ipynb thru 09_pytorch_model_deployment.ipyn.
    """)
    print("")

class ColourVGG(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1), 
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2) 
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*16*16,
                      out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x

def main():
    intro()
    my_utils.wait_for_user_input("Create a model like TinyVGG, but with colour")
    device = my_utils.get_device()
    torch.manual_seed(RANDOM_SEED)
    classes = my_utils.find_classes(directory=TRAIN_DIR)
    model = ColourVGG( input_shape=3, 
                       hidden_units=10, 
                       output_shape=len(classes) ).to(device)
    print(model)
    simple_transform = transforms.Compose([ transforms.Resize((64, 64)),transforms.ToTensor()])
    train_data_simple = datasets.ImageFolder(root=TRAIN_DIR, transform=simple_transform)
    train_dataloader_simple = DataLoader( train_data_simple, 
                                               batch_size=BATCH_SIZE, 
                                               shuffle=True, 
                                               num_workers=NUM_WORKERS)
    img_batch, label_batch = next(iter(train_dataloader_simple))

    img_single, label_single = img_batch[0].unsqueeze(dim=0), label_batch[0]
    print(f"Single image shape: {img_single.shape}\n")

    model.eval()
    with torch.inference_mode():
        pred = model(img_single.to(device))
        
    print(f"Output logits:\n{pred}\n")
    print(f"Output prediction probabilities:\n{torch.softmax(pred, dim=1)}\n")
    print(f"Output prediction label:\n{torch.argmax(torch.softmax(pred, dim=1), dim=1)}\n")
    print(f"Actual label:\n{label_single}")

if __name__ == "__main__":
    main()