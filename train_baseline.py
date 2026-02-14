import os
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from torch.utils.data import random_split
from dataset import FundusDataset
from transforms import train_transform, val_transform
from dataloader import create_loaders
from collections import Counter
import pickle
from datetime import datetime
import json

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score > self.best_score - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0
        
    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)

def main():
    # -- reproducibility -- #
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # -- folder setup -- #
    exp_name = f"baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_dir = os.path.join("experiments", exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # -- database setup -- #
    baseline_classes = ["Healthy", "Diabetic Retinopathy", "Central Serous Chorioretinopathy", "Disc Edema", "Glaucoma", "Macular Scar", "Myopia", "Retinal Detachment", "Retinitis Pigmentosa"]

    dataset = FundusDataset(
        root_dir=r"data/Original_Dataset",
        transform=None,
        class_filter=baseline_classes
    )

    # train/test/validation split
    train_size = int(0.75 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=generator)

    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    test_dataset.dataset.transform = val_transform

    print(f"""
train size: {len(train_dataset)}
val size: {len(val_dataset)}
test size: {len(test_dataset)}
total: {len(train_dataset) + len(val_dataset) + len(test_dataset)} 
        """)

    # saving validation indices for eval
    torch.save(val_dataset.indices, os.path.join(exp_dir, "val_indices.pt"))

    train_loader, val_loader, test_loader = create_loaders(train_dataset, val_dataset, test_dataset, batch_size=128)


    # -- simple CNN definition -- #
    class simpleCNN(nn.Module):
        def __init__(self, num_classes=9):
            super(simpleCNN, self).__init__()

            self.conv_layers = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )

            self.global_pool = nn.AdaptiveAvgPool2d((1,1))

            self.fc_layers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )
        
        def forward(self, x):
            x = self.conv_layers(x)
            x = self.global_pool(x)
            return self.fc_layers(x)
        

    # -- training setup -- #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = simpleCNN(num_classes=len(baseline_classes)).to(device)

    # weighted loss criterion to handle class imbalance
    labels_list = [label for _, label in dataset.samples]
    class_counts = Counter(labels_list)

    print("\noriginal class counts")
    for i, class_name in enumerate(baseline_classes):
        print(f"{class_name}: {class_counts[i]}")
    print(f"total images: {len(dataset)}\n")

    class_weights = torch.tensor(
        [1.0 / class_counts[i] for i in range(len(baseline_classes))],
        dtype=torch.float
    )
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimiser = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimiser, step_size=30, gamma=0.1)
    early_stopping = EarlyStopping(patience=10, delta=0.01)
    num_epochs = 100

    config = {
        "model": "simpleCNN",
        "epochs": num_epochs,
        "batch_size": 128,
        "learning_rate": 1e-3,
        "scheduler_step": 30,
        "scheduler_gamma": 0.1,
        "seed": seed,
        "classes": baseline_classes
    }
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

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

        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("early stopping triggered")
            num_epochs = epoch
            break

        # store metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"""
            Epoch {epoch+1}/{num_epochs}
            Train Loss: {train_loss:.4f}
            Train Acc: {train_acc:.4f}
            --------------
            Val Loss: {val_loss:.4f}
            Val Acc: {val_acc:.4f}
            """)

        # save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimiser.state_dict()
            }, os.path.join(exp_dir, f"checkpoint_epoch_{epoch+1}.pth"))
            print(f"checkpoint saved at epoch {epoch+1}")
        
        scheduler.step()

    early_stopping.load_best_model(model)
    torch.save(model.state_dict(), os.path.join(exp_dir, "best_model.pth"))
    print("training complete")
        
    # -- save model -- #
    torch.save(model.state_dict(), os.path.join(exp_dir, "final_model.pth"))
    print("model saved as final_model.pth")

    # -- plotting loss curves -- #
    plt.figure(figsize=(8,6))
    plt.plot(range(1, num_epochs+1), train_losses, label="Train Loss")
    plt.plot(range(1, num_epochs+1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "loss_curves.png"))
    plt.show()

    # -- plotting accuracy curves -- #
    plt.figure(figsize=(8,6))
    plt.plot(range(1, num_epochs+1), train_accs, label="Train Accuracy")
    plt.plot(range(1, num_epochs+1), val_accs, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "accuracy_curves.png"))
    plt.show()

    # -- saving metrics arrays -- #
    with open(os.path.join(exp_dir, "metrics.pkl"), "wb") as f:
        pickle.dump({
            "train_losses": train_losses,
            "train_accs": train_accs,
            "val_losses": val_losses,
            "val_accs": val_accs
        }, f)

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()


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
