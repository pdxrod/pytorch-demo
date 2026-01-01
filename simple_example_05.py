#!/usr/bin/env python

from imports import *

DATA_PATH = Path("data/")
IMAGE_PATH = DATA_PATH / "pizza_steak_sushi"
TRAIN_DIR = IMAGE_PATH / "train"
TEST_DIR = IMAGE_PATH / "test"
BATCH_SIZE = 32
RANDOM_SEED = 42
NUM_WORKERS = os.cpu_count()
NUM_EPOCHS = 5
LEARNING_RATE = 0.001

def intro():
    print("")
    my_utils.pretty_print("""
This program is intended to summarize the notebooks pytorch-deep-learning/ 
04_pytorch_custom_datasets.ipynb thru 09_pytorch_model_deployment.ipyn.
    """)
    print("")

class TinyVGG(nn.Module):
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
    my_utils.wait_for_user_input("Create a model and train it:")
    device = my_utils.get_device()
    torch.manual_seed(RANDOM_SEED)
    classes = my_utils.find_classes(directory=TRAIN_DIR)
    model = TinyVGG( input_shape=3, 
                       hidden_units=10, 
                       output_shape=len(classes) ).to(device)
    print(model)
    
    simple_transform = transforms.Compose([ transforms.Resize((64, 64)),transforms.ToTensor()])
    train_data_simple = datasets.ImageFolder(root=TRAIN_DIR, transform=simple_transform)
    train_dataloader_simple = DataLoader( train_data_simple, 
                                               batch_size=BATCH_SIZE, 
                                               shuffle=True, 
                                               num_workers=NUM_WORKERS)
    
    train_transform_trivial = transforms.Compose([ transforms.Resize(size=(64, 64)),
                                                   transforms.TrivialAugmentWide(num_magnitude_bins=31),
                                                   transforms.ToTensor() ])
    train_data_augmented = datasets.ImageFolder(root=TRAIN_DIR,
                                                transform=train_transform_trivial)
    train_dataloader_augmented = DataLoader(dataset=train_data_augmented,
                                            batch_size=BATCH_SIZE,
                                            shuffle=True,
                                            num_workers=NUM_WORKERS)
    test_transform_simple = transforms.Compose([ transforms.Resize(size=(64, 64)),
                                                 transforms.ToTensor() ])
    test_data_simple = datasets.ImageFolder(root=TEST_DIR,
                                            transform=test_transform_simple)
    test_dataloader_simple = DataLoader(dataset=test_data_simple,
                                        batch_size=BATCH_SIZE,
                                        shuffle=False,
                                        num_workers=NUM_WORKERS)
    
    img_batch, label_batch = next(iter(train_dataloader_simple))

    img_single, label_single = img_batch[0].unsqueeze(dim=0), label_batch[0]
    print(f"Single image shape: {img_single.shape}\n")

    model.eval()
    with torch.inference_mode():
        pred = model(img_single.to(device)).item()
        
    print(f"Output logits:\n{pred}\n")
    pred_probs = torch.softmax(pred, dim=1).item()
    print(f"Output prediction probabilities:\n{pred_probs}\n")
    pred_label = torch.argmax(pred_probs, dim=1).item()
    print(f"Output prediction label:\n{pred_label}")
    print(f"Actual label:\n{label_single} {label_single.item()}")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    results = my_utils.train_loop( model=model,
                                   train_dataloader=train_dataloader_augmented,
                                   test_dataloader=test_dataloader_simple,
                                   optimizer=optimizer,
                                   loss_fn=loss_fn,
                                   epochs=NUM_EPOCHS,
                                   device=device)
    my_utils.plot_loss_curves(results)

if __name__ == "__main__":
    main()