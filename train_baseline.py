import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from torch.utils.data import random_split, DataLoader
from dataset import FundusDataset
from transforms import train_transform
import matplotlib.pyplot as plt
import numpy as np
import random

# -- reproducibility -- #
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

# -- database setup -- #
baseline_classes = ["Healthy", "Diabetic Retinopathy", "Central Serous Chorioretinopathy", "Disc Edema", "Glaucoma", "Macular Scar", "Myopia", "Retinal Detachment", "Retinitis Pigmentosa"]

dataset = FundusDataset(
    root_dir=r"data/Augmented_Dataset",
    transform=train_transform,
    class_filter=baseline_classes
)

# train/validation split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
generator = torch.Generator().manual_seed(seed)
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

# saving validation indices for eval
torch.save(val_dataset.indices, "val_indices.pt")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# -- simple CNN definition -- #
class simpleCNN(nn.Module):
    def __init__(self, num_classes=9):
        super(simpleCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*28*28, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        return self.fc_layers(x)
    

# -- training setup -- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = simpleCNN(num_classes=len(baseline_classes)).to(device)

criterion = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 5

# -- training loop -- #
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels, _ in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimiser.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimiser.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    train_loss = running_loss / total
    train_acc = correct / total

    # validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels, _ in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss /= val_total
    val_acc = val_correct / val_total

    print(f"""
        Epoch {epoch+1}/{num_epochs}
        Train Loss: {train_loss:.4f}
        Train Acc: {train_acc:.4f}
        --------------
        Val Loss: {val_loss:.4f}
        Val Acc: {val_acc:.4f}
          """)
    
# -- save model -- #
torch.save(model.state_dict(), "baseline_cnn.pth")
print("model saved as baseline_cnn.pth")

"""
# -- visualise predictions on 4 validation images -- #
model.eval()
images, labels, class_names = next(iter(val_loader))
images, labels = images.to(device), labels.to(device)

# get predictions
with torch.no_grad():
    outputs = model(images)
    probs = f.softmax(outputs, dim=1)
    preds = torch.argmax(probs, dim=1)

images = images.cpu()
labels = labels.cpu()
preds = preds.cpu()

def unnormalise(img):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    return img * std + mean

for i in range(4):
    img = unnormalise(images[i]).permute(1, 2, 0)
    plt.imshow(img)
    plt.title(f"True: {labels[i].item()} | Pred: {preds[i].item()} | Class: {class_names[i]}")
    plt.axis("off")
    plt.show()
"""