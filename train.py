import torch
from torch.utils.data import random_split, DataLoader
from dataset import FundusDataset
from transforms import train_transform
import matplotlib.pyplot as plt

def unnormalise(img):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    return img * std + mean

baseline_classes = ["Healthy", "Diabetic Retinopathy"]

dataset = FundusDataset(
    root_dir=r"data/Original_Dataset",
    transform=train_transform,
    class_filter=baseline_classes
)

# train / validation split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

# sanity check
images, labels, class_names = next(iter(train_loader))
print(f"Shape: {images.shape}\nLabels: {labels}\nClass: {class_names}")

for i in range(4):
    img = unnormalise(images[i])
    img = img.permute(1, 2, 0)
    plt.imshow(img)
    plt.title(f"Label: {labels[i].item()}, Class: {class_names[i]}")
    plt.axis("off")
    plt.show()