#!/usr/bin/env python

from imports import *
import torch.nn.functional as F

DATA_PATH = Path("data/")
IMAGE_PATH = DATA_PATH / "pizza_steak_sushi"
TRAIN_DIR = IMAGE_PATH / "train"
TEST_DIR = IMAGE_PATH / "test"
CUSTOM_IMAGE_PATH = DATA_PATH / "04-pizza-dad.jpeg"
BATCH_SIZE = 32
RANDOM_SEED = 42
NUM_WORKERS = 0  # Use 0 on macOS to avoid multiprocessing issues
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

def create_dataloaders(train_dir: str,
                       test_dir: str,
                       batch_size: int = 32,
                       num_workers: int = None,
                       image_size: tuple = (64, 64)):
    if num_workers is None:
        num_workers = os.cpu_count()
    
    simple_transform = transforms.Compose([ transforms.Resize(image_size), transforms.ToTensor() ])
    train_data_simple = datasets.ImageFolder(root=train_dir, transform=simple_transform)
    train_dataloader_simple = DataLoader(
        train_data_simple,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    train_transform_trivial = transforms.Compose([ transforms.Resize(size=image_size), 
                                                   transforms.TrivialAugmentWide(num_magnitude_bins=31), 
                                                   transforms.ToTensor() ])
    train_data_augmented = datasets.ImageFolder(root=train_dir, transform=train_transform_trivial)
    train_dataloader_augmented = DataLoader(
        dataset=train_data_augmented,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    test_transform_simple = transforms.Compose([
        transforms.Resize(size=image_size),
        transforms.ToTensor()
    ])
    test_data_simple = datasets.ImageFolder(root=test_dir, transform=test_transform_simple)
    test_dataloader_simple = DataLoader(
        dataset=test_data_simple,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_dataloader_simple, train_dataloader_augmented, test_dataloader_simple

def main():
    intro()
    my_utils.wait_for_user_input("Create models and train them:")
    device = my_utils.get_device()
    torch.manual_seed(RANDOM_SEED)
    classes = my_utils.find_classes(directory=TRAIN_DIR)
    train_transform_trivial = transforms.Compose([ transforms.Resize(size=(64, 64)),
                                                   transforms.TrivialAugmentWide(num_magnitude_bins=31),
                                                   transforms.ToTensor() ])
    train_data_augmented = datasets.ImageFolder(root=TRAIN_DIR,
                                            transform=train_transform_trivial)

    train_dataloader_simple, train_dataloader_augmented, test_dataloader_simple = create_dataloaders(
        train_dir=TRAIN_DIR,
        test_dir=TEST_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        image_size=(64, 64) )
    model_0 = TinyVGG( input_shape=3, 
                       hidden_units=10, 
                       output_shape=len(classes) ).to(device)
    model_1 = TinyVGG(input_shape=3,
                      hidden_units=10,
                      output_shape=len(train_data_augmented.classes)).to(device)
    print("model_0:")
    print(model_0)
    print("model_1:")
    print(model_1)
    print("\nWait...")                   
    
    img_batch, label_batch = next(iter(train_dataloader_simple))

    img_single, label_single = img_batch[0].unsqueeze(dim=0), label_batch[0]

    model_0.eval()
    with torch.inference_mode():
        pred = model_0(img_single.to(device))

    custom_image = torchvision.io.read_image(str(CUSTOM_IMAGE_PATH)).type(torch.float32) / 255.
    model_1.eval()
    with torch.inference_mode():
        image_resized = F.interpolate(
            custom_image.unsqueeze(0), 
            size=(64, 64),
            mode='bilinear',
            align_corners=False
        )
        pred = model_1(image_resized.to(device))
        values = pred.squeeze().to(device).tolist()            
        print(f"Output logits:\n{values}\n")
        pred_probs = torch.softmax(pred, dim=1)
        values = pred_probs.squeeze().to(device).tolist()            
        print(f"Output prediction probabilities:\n{values}\n")
        pred_label = torch.argmax(pred_probs, dim=1).item()
        print(f"Output prediction label:\n{pred_label}")
        print(f"Actual label:\n{label_single}")
        print("Training loops for model_0 and model_1...")

    loss_fn = nn.CrossEntropyLoss()

    print("Model 0 has been trained on simple data.")
    optimizer_0 = torch.optim.Adam(params=model_0.parameters(), lr=LEARNING_RATE)
    model_0_results = my_utils.train_loop( model=model_0,
                                           train_dataloader=train_dataloader_simple,
                                           test_dataloader=test_dataloader_simple,
                                           optimizer=optimizer_0,
                                           loss_fn=loss_fn,
                                           epochs=NUM_EPOCHS,
                                           device=device )
    print("Model 1 has been trained on data augemented with TrivialAugmentWide.")
    optimizer_1 = torch.optim.Adam(params=model_1.parameters(), lr=LEARNING_RATE)
    model_1_results = my_utils.train_loop( model=model_1,
                                           train_dataloader=train_dataloader_augmented,
                                           test_dataloader=test_dataloader_simple,
                                           optimizer=optimizer_1,
                                           loss_fn=loss_fn,
                                           epochs=NUM_EPOCHS,
                                           device=device )

    print("Show loss curves...")
    my_utils.wait_for_user_input("Model 0 loss curves:")
    my_utils.plot_loss_curves(model_0_results)
    plt.show()
    my_utils.wait_for_user_input("Model 1 loss curves:")
    my_utils.plot_loss_curves(model_1_results)
    plt.show()

if __name__ == "__main__":
    main()