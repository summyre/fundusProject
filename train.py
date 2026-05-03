import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset
from dataset import FundusDataset, create_loaders, TransformDataset, train_transform, val_transform
from functions import plot_history, evaluate_model, generate_gradcam
from models import resnet18, set_trainable_layers
from sklearn.metrics import f1_score
import numpy as np
import time
import random
import os
import optuna
from collections import Counter

num_epochs = 100        # max number of training epochs
tuning = False          # disable/enable hyperparameter optimisation

# list of diease classes used in classification task
baseline_classes = [
    "Healthy", "Diabetic Retinopathy", "Central Serous Chorioretinopathy", "Disc Edema", "Glaucoma", 
    "Macular Scar", "Myopia", "Retinal Detachment", "Retinitis Pigmentosa"
]

# -- define training loop -- #
def train_model(model, train_loader, val_loader, criterion, optimiser, device, num_epochs, scheduler, name, trial=None):
    model.to(device)
    # early stopping configuration
    patience = 5                    # number of epochs wihtout improvement before stopping
    no_improve = 0
    best_loss = float("inf")

    # storing training history for later visualisation
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    # epoch loop
    for epoch in range(num_epochs): # loop over the dataset multiple times
        start_time = time.time()
        # tracking metrics per epoch
        train_loss, train_correct, train_total = 0.0, 0, 0
        val_loss, val_correct, val_total = 0.0, 0, 0

        # training
        model.train()
        for data, target, _ in train_loader:
            data, target = data.to(device), target.to(device)   # get the inputs
            optimiser.zero_grad()               # set the parameter gradients to zero
            output = model(data)                # forward pass
            loss = criterion(output, target)    # compute loss
            loss.backward()                     # backpropagation
            optimiser.step()                    # update weights
            
            # totalling loss weighted by batch size
            train_loss += loss.item() * data.size(0)

            #computing training accuracy
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

        # calculating average loss
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)

        # calculating epoch metrics
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        # optuna pruning (for tuning only)
        if trial is not None:
            trial.report(val_acc, epoch)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        end_time = time.time()
        epoch_time = end_time - start_time

        print(f"Epoch: {epoch+1}/{num_epochs} | Time: {epoch_time:.3f}s | Train Acc: {train_acc:.4f} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f}")

        # saving model if val loss decreases - checkpoint
        if val_loss < best_loss:
            print(f"validation loss decreased ({best_loss:.4f} -> {val_loss}). saving model as {name}.pt")
            if trial is None:
                torch.save({
                    'model': model.state_dict(),
                    'optimiser': optimiser.state_dict(),
                    'epoch': epoch
                }, f'{name}.pt')
            best_loss = val_loss
            no_improve = 0
        else:
            no_improve += 1

        # early stopping condition
        if no_improve >= patience:
            print("early stopping triggered")
            break

        scheduler.step()

        # saving metrics
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
    
    return history

# -- experiment runner -- #
def run(params, train_dataset, val_dataset, test_dataset, device, class_weights, trial=None):
    lr = params["lr"]
    num_classes = len(baseline_classes)

    # creating dataloaders
    train_loader, val_loader, _ = create_loaders(train_dataset, val_dataset, test_dataset, batch_size=128)

    # loading pretrained resnet18 model
    model = resnet18(num_classes, pretrained=True, dropout=params["dropout"])
    model.to(device)

    # applying freezing strategy 
    model = set_trainable_layers(model, params["freeze_mode"])
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())     # only optimise trainable params

    # selecting an optimiser
    if params["optimiser"] == "adam":
        optimiser = optim.Adam(trainable_params, lr=lr, weight_decay=params["weight_decay"])
    else:
        optimiser = optim.SGD(trainable_params, lr=lr, momentum=0.9)

    # learning rate schedule
    scheduler = optim.lr_scheduler.StepLR(optimiser, step_size=5, gamma=0.1)
    # weighted loss to handle class imbalance
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    train_model(model, train_loader, val_loader, criterion, optimiser, device, num_epochs=15, scheduler=scheduler, name="temp", trial=trial)

    # evaluate macro f1
    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for data, target, _ in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            preds = torch.argmax(output, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    return f1_score(all_labels, all_preds, average="macro")

def main():
    # -- reproducibility -- #
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # loading fundus image dataset and filtering to selected disease classes
    dataset = FundusDataset(
        root_dir=r"data/Original_Dataset",
        transform=None,
        class_filter=baseline_classes
    )
    num_classes = len(baseline_classes)

    # train-validation-test split
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)               # randomly shuffle dataset indices before splitting

    # split 75/15/10
    train_size = int(0.75 * len(indices))
    val_size = int(0.15 * len(indices))

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]

    # applying data transformations
    train_dataset = TransformDataset(Subset(dataset, train_indices), transform=train_transform)    # augmentation
    val_dataset = TransformDataset(Subset(dataset, val_indices), transform=val_transform)          # deterministic preprocessing
    test_dataset = TransformDataset(Subset(dataset, test_indices), transform=val_transform)

    # creating dataloaders
    train_loader, val_loader, test_loader = create_loaders(train_dataset, val_dataset, test_dataset, batch_size=128)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        # use gpu acceleration if CUDA is available

    # weighted loss criterion to handle class imbalance
    train_labels = []
    
    for _, labels, _ in train_loader:
        train_labels.extend(labels.cpu().numpy())
    
    class_counts = Counter(train_labels)

    print("training class distribution: ")
    for i, class_name in enumerate(baseline_classes):
        print(f"{class_name}: {class_counts[i]}")

    total_samples = len(train_labels)
    # computing class weights inversely proportional to class frequency
    class_weights = torch.tensor(
        [np.sqrt(total_samples / (num_classes * class_counts.get(i, 1))) for i in range(num_classes)],
        dtype=torch.float
    )

    # preventing excessively large weights
    class_weights = torch.clamp(class_weights, max=2.5)
    class_weights = class_weights.to(device)

    print(class_weights)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # -- hyperparameter optimisation -- #
    if tuning:
        def objective(trial):
            params = {
                "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
                "dropout": trial.suggest_float("dropout", 0.3, 0.7),
                "optimiser": trial.suggest_categorical("optimiser", ["adam", "sgd"]),
                "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
                "freeze_mode": trial.suggest_categorical("freeze_mode", ["full", "freeze", "partial"])
            }

            return run(params, train_dataset, val_dataset, test_dataset, device, class_weights, trial)

        # hyperparameter tuning with fixed class imbalance handling
        print("\nstarting hyperparameter tuning\n")
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=20)

        print("\nbest trial:")
        print(f"value: {study.best_value}")
        print(f"params: {study.best_params}")

        best_params = study.best_params

        # -- build resnet18 using best hyperparameters -- #
        rn_model = resnet18(num_classes, pretrained=True, dropout=best_params["dropout"]).to(device)
        freeze_mode = best_params.get("freeze_mode", "full")
        rn_model = set_trainable_layers(rn_model, mode=freeze_mode)
        trainable_params = filter(lambda p: p.requires_grad, rn_model.parameters())

        if best_params["optimiser"] == "adam":
            rn_optim = torch.optim.Adam(trainable_params, lr=best_params["lr"], weight_decay=best_params["weight_decay"])
        else:
            rn_optim = optim.SGD(trainable_params, lr=best_params["lr"], momentum=0.9)
    else:
        # default training configuration / current best hyperparam after tuning
        rn_model = resnet18(num_classes, pretrained=True, dropout=0.3).to(device)
        rn_model = set_trainable_layers(rn_model, mode="full")                  # change to freeze or partial to test
        trainable_params = filter(lambda p: p.requires_grad, rn_model.parameters())
        rn_optim = torch.optim.Adam(trainable_params, lr=2e-4, weight_decay=1e-5)

    rn_scheduler = optim.lr_scheduler.StepLR(rn_optim, step_size=30, gamma=0.1)

    history_rn = train_model(rn_model, train_loader, val_loader, criterion, rn_optim, device, num_epochs, rn_scheduler, "resnet18")

    # load best model
    ckp = torch.load('resnet18.pt')
    rn_model.load_state_dict(ckp['model'])
    print(f"best epoch: {ckp['epoch']}")
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
    print(f"resnet18 test loss: {test_loss:.4f}")

    overall_acc = np.sum(class_correct) / np.sum(class_total)
    print(f"\nresnet18 test accuracy (overall): {overall_acc:.4f}")

    evaluate_model(rn_model, test_loader, device, baseline_classes, exp_dir="resnet18", split_name="test")

    # generating Grad-CAM visualisations - used to interpret model attention regions
    generate_gradcam(rn_model, test_loader, device, exp_dir="resnet18/gradcam", class_names=baseline_classes, num_images=50)

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
