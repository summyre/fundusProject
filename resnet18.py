import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from torch.utils.data import random_split, Subset
import torchvision.transforms as transforms
import matplotlib as plt
import numpy as np
import random
from dataset import FundusDataset, create_loaders
from functions import EarlyStopping, evaluate_model
from collections import Counter
import datetime
import os
import json

# -- defining residual block -- #
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                                          nn.BatchNorm2d(out_channels))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out
    
class ResNet18(nn.Module):
    def __init__(self, num_classes=9):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.flatten(out, 1)
        out = self.fc(out)
        return out

def main():
    # -- reproducibility -- #
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # -- folder setup -- #
    exp_name = f"resnet18_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_dir = os.path.join("experiments", exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # -- load and preprocess dataset -- #
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        )
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        )
    ])

    baseline_classes = ["Healthy", "Diabetic Retinopathy", "Central Serous Chorioretinopathy", "Disc Edema", "Glaucoma", "Macular Scar", "Myopia", "Retinal Detachment", "Retinitis Pigmentosa"]

    dataset = FundusDataset(
            root_dir=r"data/Original_Dataset",
            transform=None,
            class_filter=baseline_classes
        )

    train_size = int(0.75 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    generator = torch.Generator().manual_seed(seed)
    train_indices, val_indices, test_indices = random_split(range(len(dataset)), [train_size, val_size, test_size], generator=generator)

    train_dataset = FundusDataset(
        root_dir=r"data/Original_Dataset",
        transform=transform_train,
        class_filter=baseline_classes
    )
    val_dataset = FundusDataset(
        root_dir=r"data/Original_Dataset",
        transform=transform_test,
        class_filter=baseline_classes
    )
    test_dataset = FundusDataset(
        root_dir=r"data/Original_Dataset",
        transform=transform_test,
        class_filter=baseline_classes
    )

    train_dataset = Subset(train_dataset, train_indices.indices)
    val_dataset = Subset(val_dataset, val_indices.indices)
    test_dataset = Subset(test_dataset, test_indices.indices)

    train_loader, val_loader, test_loader = create_loaders(train_dataset, val_dataset, test_dataset, batch_size=128)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18(num_classes=len(baseline_classes)).to(device)

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

    # -- defining loss function and optimiser --#
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimiser = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimiser, step_size=30, gamma=0.1)
    early_stopping = EarlyStopping(patience=5)
    num_epochs = 100

    config = {
        "model": "ResNet18",
        "epochs": num_epochs,
        "batch_size": 128,
        "learning_rate": 0.01,
        "scheduler_step": 30,
        "scheduler_gamma": 0.1,
        "seed": seed,
        "classes": baseline_classes
    }
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)


    # -- training the model -- #

    train_losses , train_accs, test_accs = [], [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        for inputs, labels, _ in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimiser.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()

            running_loss += loss.item() * inputs.size(0)
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
        
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = 100. * correct / total
        val_acc, val_f1, val_recall = evaluate_model(model, val_loader, device, baseline_classes, exp_dir, split_name="val")

        early_stopping(train_loss, model)
        if early_stopping.early_stop:
            print("early stopping triggered")
            break

        # store metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels, _ in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, pred = outputs.max(1)
                total += labels.size(0)
                correct += pred.eq(labels).sum().item()
        
        test_acc = 100. * correct / total
        test_accs.append(test_acc)

        # save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimiser.state_dict()
            }, os.path.join(exp_dir, f"checkpoint_epoch_{epoch+1}.pth"))
            print(f"checkpoint saved at epoch {epoch+1}")

        scheduler.step()
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

    early_stopping.load_best_model(model)
    test_acc, test_f1, test_recall = evaluate_model(model, test_loader, device, baseline_classes, exp_dir, split_name="test")
    epochs_ran = len(train_losses)
    torch.save(model.state_dict(), os.path.join(exp_dir, "best_model.pth"))

    # -- plotting training loss and accuracy -- #
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.savefig(os.path.join(exp_dir, "loss_curves.png"))

    plt.figure(figsize=(8,6))
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(exp_dir, "accuracy_curves.png"))

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()