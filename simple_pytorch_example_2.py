#!/usr/bin/env python
 
"""
Get optimizer function
for N epochs:
    train step:
      model.train()
      get loss function
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    test step:
      model.eval()
      test_loss = loss_fn(...)
evaluate test loss            
"""

from imports import *
import my_utils

DEVICE = my_utils.get_device()
EPOCHS = 1000
SAMPLES = 1000
NOISE = 0.03
RANDOM_SEED = 42
TEST_SIZE = 0.2
LEARNING_RATE = 0.1
SNAPSHOT_EPOCHS = [0, 50, 200, 999]  # 4 snapshots → 8 images total (2 models x 4)

my_utils.pretty_print("""
This example has two PyTorch models.
The program illustrates the use of an optimizer and a loss function to 
improve the ability of models to find boundaries.
It generates two half-moon shaped objects on a two-dimensional 
plane. These objects overlap, so it is impossible to draw a 
straight line between them which does not bump into either of the half-moons.
The first model is only able to draw straight lines.
""")
print(""" 
The second model, with the addition of a function called 'ReLU', 
    self.layer1 = nn.Linear(in_features=2, out_features=10)
    self.relu1 = nn.ReLU()  
in the second model's constructor, and
    x = self.relu1(x)
in its forward() function, is able to work out how to draw a 
curved line which does not hit either object.   """)
my_utils.wait_for_user_input("""The difference between the two models is illustrated, first by showing the numbers in
their attempts to reduce 'loss', then by a series of images showing the results.   
""")

class ClassifierModel1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=2, out_features=10)
        self.layer2 = nn.Linear(in_features=10, out_features=10)
        self.layer3 = nn.Linear(in_features=10, out_features=1)

    def forward(self, x):
        return self.layer3( self.layer2( self.layer1(x)))

class ClassifierModel2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=2, out_features=10)
        self.relu1 = nn.ReLU()  
        self.layer2 = nn.Linear(in_features=10, out_features=10)
        self.relu2 = nn.ReLU() 
        self.layer3 = nn.Linear(in_features=10, out_features=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)      
        x = self.layer2(x)
        x = self.relu2(x)      
        x = self.layer3(x)
        return x        

def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor, ax=None):
    """
    Source - https://madewithml.com/courses/foundations/neural-networks/ (with modifications)
    """
    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # mutli-class
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))  # binary

    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    
    # Use provided axes or default to plt
    if ax is None:
        ax = plt
    
    ax.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    ax.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() 
    acc = (correct / len(y_pred)) * 100 
    return acc

model_1 = ClassifierModel1().to(DEVICE)
model_2 = ClassifierModel2().to(DEVICE)
models = [model_1, model_2]

# Get X and y for 'moons' data 
# X = input features (coordinates of each point)
# y = target labels (which class each point belongs to: 0 or 1)
X, y = make_moons(n_samples=SAMPLES, noise=NOISE, random_state=RANDOM_SEED)

loss_fn = nn.BCEWithLogitsLoss()

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=TEST_SIZE, 
                                                     random_state=RANDOM_SEED )
X_train_gpu_or_mps, X_test_gpu_or_mps = X_train.to(DEVICE), X_test.to( DEVICE )
y_train_gpu_or_mps, y_test_gpu_or_mps = y_train.to(DEVICE), y_test.to( DEVICE )
y_train_float = y_train_gpu_or_mps.float()
y_train_int = y_train_gpu_or_mps.long()
y_test_float = y_test_gpu_or_mps.float()
y_test_int = y_test_gpu_or_mps.long()

print("Training loop")
n = 1
# 1. Forward pass:     X_train → model → y_logits (predictions)
# 2. Calculate loss:   Compare y_logits to y_train → loss (how wrong?)
# 3. Backward pass:    loss.backward() → gradients (which way to adjust?)
# 4. Update weights:   optimizer.step() → new weights (actually adjust)
# 5. Repeat:           Do this 1000 times → model learns!
# 6. Test:             Check on unseen data → see if it generalizes
#   loss & acc - How well the model performs on data it's learning from.
#   test_loss & test_acc - How well the model generalizes to unseen data.

# Store snapshots for each model (for end-of-run comparison grid)
model_snapshots = []

# Capture snapshots during training to show boundary progression
model_factories = [ClassifierModel1, ClassifierModel2]

for model_idx, model in enumerate(models):
    print("")
    print(f"model_{n}")
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    snapshots = []
    for epoch in range(EPOCHS):
        model.train()
        y_logits = model(X_train_gpu_or_mps)
        loss = loss_fn(y_logits.squeeze(), y_train_gpu_or_mps.squeeze())
        y_pred = torch.round(torch.sigmoid(y_logits)).squeeze()
        acc = accuracy_fn(y_true=y_train_int, y_pred=y_pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            test_logits = model(X_test_gpu_or_mps)
            test_pred = torch.round(torch.sigmoid(test_logits)).squeeze()
            test_loss = loss_fn(test_logits.squeeze(), y_test_gpu_or_mps.squeeze())
            test_acc = accuracy_fn(y_true=y_test_int, y_pred=test_pred)
            if (epoch < 100 and epoch % 20 == 0) or (epoch > 100 and epoch % 100 == 0):
                print(f"Epoch {epoch:04d} | test_loss: {test_loss:.4f} |\
 test_acc: {test_acc:.2f}% | loss: {loss:.4f} | acc: {acc:.2f}%")

            # Take snapshots at selected epochs to visualize later
            if epoch in SNAPSHOT_EPOCHS or epoch == EPOCHS - 1:
                snapshots.append({
                    "epoch": epoch,
                    "state_dict": deepcopy(model.state_dict()),
                    "test_loss": float(test_loss.item()),
                    "test_acc": float(test_acc)
                })
    print(f"Final for model_{n}: Test loss: {test_loss:.4f} | Test acc: {test_acc:.2f}%")
    model_snapshots.append({
        "model_num": n,
        "snapshots": snapshots
    })
    n += 1

# Now display both models side by side for comparison
print("")
print("Diagrams")

# Build a grid: rows = models, cols = snapshot epochs
cols = len(SNAPSHOT_EPOCHS)
fig, axes = plt.subplots(len(models), cols, figsize=(4 * cols, 4 * len(models)))

for row_idx, model_info in enumerate(model_snapshots):
    model_num = model_info["model_num"]
    for col_idx, snap_epoch in enumerate(SNAPSHOT_EPOCHS):
        ax = axes[row_idx, col_idx] if len(models) > 1 else axes[col_idx]
        # Find the snapshot for this epoch (if missing, skip)
        snap = next((s for s in model_info["snapshots"] if s["epoch"] == snap_epoch), None)
        if snap is None:
            ax.axis("off")
            continue
        # Recreate model at this snapshot
        temp_model = model_factories[row_idx]().to("cpu")
        temp_model.load_state_dict(snap["state_dict"])
        plot_decision_boundary(temp_model, X_test, y_test, ax=ax)
        ax.set_title(
            f"Model {model_num} | epoch {snap_epoch}\n"
            f"acc: {snap['test_acc']:.2f}%  loss: {snap['test_loss']:.4f}",
            fontsize=10
        )

plt.suptitle("Decision Boundary Progression", fontsize=14)
plt.tight_layout()
plt.show()
