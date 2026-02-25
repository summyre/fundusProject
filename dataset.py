from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from PIL import Image
import pandas as pd
import os
import cv2
import numpy as np

# -- dataset class -- #
class FundusDataset(Dataset):
    def __init__(self, root_dir, transform=None, class_filter=None, enhance=False):
        self.samples = []
        self.class_to_idx = {}
        self.transform = transform
        self.enhance = enhance

        classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

        if class_filter:
            classes = [c for c in class_filter if c in classes]

        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

        for cls_name in classes:
            cls_path = os.path.join(root_dir, cls_name)
            for fname in os.listdir(cls_path):
                if fname.lower().endswith((".jpg")):
                    self.samples.append((os.path.join(cls_path, fname), self.class_to_idx[cls_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        class_name = os.path.basename(os.path.dirname(img_path))
        image = Image.open(img_path).convert("RGB")

        # enhancements
        if self.enhance:
            image = apply_clahe(image)

        # transforms
        if self.transform:
            image = self.transform(image)

        return image, label, class_name

# -- enhancements -- #
# CLAHE function
def apply_clahe(img):
    img = np.array(img)

    # convert to lab colour space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    # merge back
    merged = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_Lab2BGR)

    return Image.fromarray(enhanced)


# -- dataloaders -- #
def create_loaders(train_dataset, val_dataset, test_dataset, batch_size=32, pin_memory=True):
    if pin_memory and not torch.cuda.is_available():
        pin_memory = False
        print("cuda is not available, setting pin_memory=False")
        
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_dataset

# -- transforms -- #
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
    )
])