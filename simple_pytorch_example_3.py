#!/usr/bin/env python

from imports import *

EPOCHS = 100
SAMPLES = 1000
NOISE = 0.03
RANDOM_SEED = 42
TEST_SIZE = 0.2
LEARNING_RATE = 0.1
BATCH_SIZE = 32

torch.manual_seed(RANDOM_SEED)

my_utils.pretty_print("""
An important example from AI history.
Grainy pictures of clothing from the Fashion MNIST dataset.
The Fashion MNIST dataset contains 70,000 images of 10 different categories of clothing.
Each image is a 28x28 grayscale image.
The dataset is split into 60,000 training images and 10,000 test images.
The images are labeled with the category of clothing they represent.
The dataset is used to train a model to be able to identify a picture of clothing 
which is in one of the 10 categories, but which it has not seen before. 
""")
train_data = datasets.FashionMNIST(
    root="data", 
    train=True, # get training data
    download=True, 
    transform=ToTensor(), 
    target_transform=None 
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False, # get test data
    download=True,
    transform=ToTensor()
)

# Turn datasets into iterables (batches)
train_dataloader = DataLoader(train_data, 
    batch_size=BATCH_SIZE, 
    shuffle=True 
)

test_dataloader = DataLoader(test_data,
    batch_size=BATCH_SIZE,
    shuffle=False 
)

my_utils.wait_for_user_input( f"train data: {train_data}" )

# Show one example of each clothing category in a single grid image
print(f"{'='*60}")
print(f"Total classes: {len(train_data.classes)}")
print(f"Class names: {train_data.classes}\n")

# Find one example of each category
examples = {}  # {label: image}
for i in range(len(train_data)):
    image, label = train_data[i]
    if label not in examples:
        examples[label] = image
        if len(examples) == len(train_data.classes):
            break

# Create a grid showing all categories in one image
num_classes = len(train_data.classes)
rows = 2
cols = 5
fig, axes = plt.subplots(rows, cols, figsize=(12, 6))
fig.suptitle('FashionMNIST - All Categories', fontsize=16)

for idx, label in enumerate(sorted(examples.keys())):
    row = idx // cols
    col = idx % cols
    image = examples[label]
    class_name = train_data.classes[label]
    
    # Squeeze to remove channel dimension: [1, 28, 28] -> [28, 28]
    axes[row, col].imshow(image.squeeze(), cmap='gray')
    axes[row, col].set_title(f"{label}: {class_name}", fontsize=10)
    axes[row, col].axis('off')

plt.tight_layout()
plt.show(block=True)

train_features_batch, train_labels_batch = next(iter(train_dataloader))
flatten_model = nn.Flatten() # all nn modules function as a model (can do a forward pass)
x = train_features_batch[0]
output = flatten_model(x) # perform forward pass

class ClothingModel(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(in_features=input_shape, out_features=hidden_units), 
            nn.Linear(in_features=hidden_units, out_features=output_shape)
        )
    
    def forward(self, x):
        return self.layer_stack(x)

model_0 = ClothingModel(input_shape=output.shape[1], # one for every pixel (28x28)
    hidden_units=10, 
    output_shape=len(train_data.classes) 
)
model_0.to("cpu")
loss_fn = nn.CrossEntropyLoss() 
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)

print(f"\nTraining model - wait... {EPOCHS} epochs")
results = my_utils.test_train_loop(
    model=model_0,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    accuracy_fn=accuracy_fn,
    epochs=EPOCHS
)
print(f"test train loop results: {results}")

# Understanding train_data dimensions:
# train_data is a PyTorch Dataset, not a tensor
# When you access train_data[i], you get a tuple: (image, label)
# - image has shape [1, 28, 28] = [channels, height, width] (1 channel = grayscale)
# - label is an integer (0-9 for FashionMNIST classes)
# To get the number of samples, use len(train_data), not len(train_data[0])

# Show one example from each category at the end (10 images total)
print(f"Dataset info:")
print(f"  Number of samples: {len(train_data)}")
print(f"  Image dimensions: [channels, height, width] = [1, 28, 28]")
print(f"  Classes: {train_data.classes}\n")

# Find and display one example of each category
found_labels = set()
for i in range( len(train_data) ):
    break; # Delete this line if you want to see all the pictures of clothing in colour
    image, label = train_data[i]  
    if label not in found_labels:
        class_name = train_data.classes[label]  
        my_utils.show_image(image, f"Class {label}: {class_name}", block=True)
        found_labels.add(label)
        if len(found_labels) == len(train_data.classes):
            break

