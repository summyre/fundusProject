import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset
from dataset import FundusDataset, create_loaders, TransformDataset, train_transform, val_transform
from functions import plot_history, evaluate_model
from models import Custom, resnet18
import numpy as np
import time
import random
import os
from collections import Counter

batch_size = 128
num_epochs = 100

baseline_classes = ["Healthy", "Diabetic Retinopathy", "Central Serous Chorioretinopathy", "Disc Edema", "Glaucoma", "Macular Scar", "Myopia", "Retinal Detachment", "Retinitis Pigmentosa"]

# -- define training loop -- #
def train_model(model, train_loader, val_loader, criterion, optimiser, device, num_epochs, scheduler, name):
    model.to(device)

    patience = 5
    no_improve = 0
    best_loss = float("inf")

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    for epoch in range(num_epochs): # loop over the dataset multiple times
        start_time = time.time()
        train_loss, train_correct, train_total = 0.0, 0, 0
        val_loss, val_correct, val_total = 0.0, 0, 0

        # training
        model.train()
        for data, target, _ in train_loader:
            data, target = data.to(device), target.to(device)   # get the inputs
            optimiser.zero_grad()   # set the parameter gradients to zero
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimiser.step()
            train_loss += loss.item() * data.size(0)

            _, preds = torch.max(output, 1)
            train_correct += (preds == target).sum().item()
            train_total += target.size(0)

        # validation
        model.eval()
        with torch.no_grad():
            for data, target, _ in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item() * data.size(0)

                _, preds = torch.max(output, 1)
                val_correct += (preds == target).sum().item()
                val_total += target.size(0)

        # calculate average loss
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)

        # calculate epoch metrics
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        end_time = time.time()
        epoch_time = end_time - start_time

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch: {epoch+1}/{num_epochs} | Time: {epoch_time:.3f}s | Train Acc: {train_acc:.4f} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f}")

        scheduler.step()

        # save model if val loss decreases -- early stopping
        if val_loss < best_loss:
            print(f"validation loss decreased ({best_loss:.4f} -> {val_loss}). saving model as {name}.pt")
            torch.save({
                'model': model.state_dict(),
                'optimiser': optimiser.state_dict(),
                'epoch': epoch
            }, f'{name}.pt')
            best_loss = val_loss
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print("early stopping triggered")
            break
    
    return history

def main():
    # -- reproducibility -- #
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    dataset = FundusDataset(
        root_dir=r"data/Original_Dataset",
        transform=None,
        class_filter=baseline_classes
    )
    num_classes = len(baseline_classes)
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)

    train_size = int(0.75 * len(indices))
    val_size = int(0.15 * len(indices))

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]

    train_dataset = TransformDataset(Subset(dataset, train_indices), transform=train_transform)
    val_dataset = TransformDataset(Subset(dataset, val_indices), transform=val_transform)
    test_dataset = TransformDataset(Subset(dataset, test_indices), transform=val_transform)

    train_loader, val_loader, test_loader = create_loaders(train_dataset, val_dataset, test_dataset, batch_size=128)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Custom(num_classes)

    # weighted loss criterion to handle class imbalance
    train_labels = []
    
    for _, labels, _ in train_loader:
        train_labels.extend(labels.cpu().numpy())
    
    class_counts = Counter(train_labels)

    print("training class distribution: ")
    for i, class_name in enumerate(baseline_classes):
        print(f"{class_name}: {class_counts[i]}")

    total_samples = len(train_labels)

    class_weights = torch.tensor(
        [total_samples / (num_classes * class_counts.get(i, 1)) for i in range(num_classes)],
        dtype=torch.float
    )
    
    class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimiser = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimiser, step_size=30, gamma=0.1)
    
    history_cnn = train_model(model, train_loader, val_loader, criterion, optimiser, device, num_epochs, scheduler, "custom")
    
    # load best model
    ckp = torch.load('custom.pt')
    model.load_state_dict(ckp['model'])
    print("finished training")

    os.makedirs("custom", exist_ok=True)
    plot_history(history_cnn, exp_dir="custom", model_name="customCNN")

    test_loss = 0.0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    model.eval()
    with torch.no_grad():
        for data, target, _ in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item() * data.size(0)
            _, pred = torch.max(output, 1)
            correct = pred.eq(target.view_as(pred))
            for i in range(len(target)):
                label = target[i].item()
                class_correct[label] += correct[i].item()
                class_total[label] += 1

    # print test results
    test_loss /= len(test_loader.dataset)
    print(f"customcnn test loss: {test_loss:.6f}")

    overall_acc = 100. * np.sum(class_correct) / np.sum(class_total)
    print(f"\ncustomcnn test accuracy (overall): {overall_acc:.2f}%")

    evaluate_model(model, test_loader, device, baseline_classes, exp_dir="custom", split_name="test")

    # -- resnet18 -- #
    rn_model = resnet18(num_classes, pretrained=True).to(device)
    rn_optim = torch.optim.Adam(rn_model.parameters(), lr=1e-3)
    rn_scheduler = optim.lr_scheduler.StepLR(rn_optim, step_size=30, gamma=0.1)

    history_rn = train_model(rn_model, train_loader, val_loader, criterion, rn_optim, device, num_epochs, rn_scheduler, "resnet18")

    # load best model
    ckp = torch.load('resnet18.pt')
    rn_model.load_state_dict(ckp['model'])
    print("finished training")

    os.makedirs("resnet18", exist_ok=True)
    plot_history(history_rn, exp_dir="resnet18", model_name="ResNet18")

    test_loss = 0.0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    rn_model.eval()
    with torch.no_grad():
        for data, target, _ in test_loader:
            data, target = data.to(device), target.to(device)
            output = rn_model(data)
            loss = criterion(output, target)
            test_loss += loss.item() * data.size(0)
            _, pred = torch.max(output, 1)
            correct = pred.eq(target.view_as(pred))
            for i in range(len(target)):
                label = target[i].item()
                class_correct[label] += correct[i].item()
                class_total[label] += 1

    # print test results
    test_loss /= len(test_loader.dataset)
    print(f"resnet18 test loss: {test_loss:.6f}")

    overall_acc = 100. * np.sum(class_correct) / np.sum(class_total)
    print(f"\nresnet18 test accuracy (overall): {overall_acc:.2f}%")

    evaluate_model(rn_model, test_loader, device, baseline_classes, exp_dir="resnet18", split_name="test")

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
