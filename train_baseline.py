import os
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from torch.utils.data import random_split
from dataset import FundusDataset, train_transform, val_transform, create_loaders
from collections import Counter
import pickle
from datetime import datetime
import json
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, f1_score, recall_score

# -- simple CNN definition -- #
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=9):
        super(SimpleCNN, self).__init__()

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

# -- early stopping definition -- #
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

# -- evaluate model function -- #
def evaluate_model(model, loader, device, class_names, exp_dir, split_name="val"):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # accuracy
    acc = np.mean(all_preds == all_labels)
    print(f"{split_name.upper()} accuracy = {acc:.4f}")

    # macro metrics
    m_f1 = f1_score(all_labels, all_preds, average="macro")
    m_rec = recall_score(all_labels, all_preds, average="macro")

    print(f"""
{split_name.capitalize()} Results
accuracy:       {acc:.4f}
macro f1:       {m_f1:.4f}
macro recall:   {m_rec:.4f}
""")
    
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(8,8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(xticks_rotation=45)
    plt.title(f"{split_name.capitalize()} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, f"{split_name}_confusion_matrix.png"))
    plt.show()

    # classification report
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)

    with open(os.path.join(exp_dir, f"{split_name}_classification_report.txt")):
        f.write(report)
    print(report)

    return acc, m_f1, m_rec, cm

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

    # -- training setup -- #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_classes=len(baseline_classes)).to(device)

    # weighted loss criterion to handle class imbalance
    train_labels = []
    
    for idx in train_dataset.indices:
        _, label = dataset.samples[idx]
        train_labels.append(label)
    
    class_counts = Counter(train_labels)

    print("training class distribution: ")
    for i, class_name in enumerate(baseline_classes):
        print(f"{class_name}: {class_counts[i]}")
    
    total = len(train_labels)

    class_weights = torch.tensor(
        [total / class_counts[i] for i in range(len(baseline_classes))],
        dtype=torch.float
    )

    class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimiser = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimiser, step_size=30, gamma=0.1)
    early_stopping = EarlyStopping(patience=10, delta=0.01)
    num_epochs = 100

    config = {
        "model": "SimpleCNN",
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

        if val_acc > best_val_acc:
            best_val_acc = val_acc

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
    epochs_ran = len(train_losses)
    torch.save(model.state_dict(), os.path.join(exp_dir, "best_model.pth"))

    print("evaluating on validation set")
    val_acc, val_f1, val_recall = evaluate_model(model, val_loader, device, baseline_classes, exp_dir, split_name="val")

    print("training complete")
    print(f"best validation accuracy: {best_val_acc:.4f}")
        
    # -- save model -- #
    torch.save(model.state_dict(), os.path.join(exp_dir, "final_model.pth"))
    print("model saved as final_model.pth")

    print("evaluating on test set")
    test_acc, test_f1, test_recall = evaluate_model(model, test_loader, device, baseline_classes, exp_dir, split_name="test")

    # -- plotting loss curves -- #
    plt.figure(figsize=(8,6))
    plt.plot(range(1, epochs_ran+1), train_losses, label="Train Loss")
    plt.plot(range(1, epochs_ran+1), val_losses, label="Validation Loss")
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
    plt.plot(range(1, epochs_ran+1), train_accs, label="Train Accuracy")
    plt.plot(range(1, epochs_ran+1), val_accs, label="Validation Accuracy")
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
