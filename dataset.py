from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from PIL import Image
import os
import cv2
import numpy as np

# -- dataset class -- #
class FundusDataset(Dataset):
    # initialising dataset attributes
    def __init__(self, root_dir, transform=None, class_filter=None):
        self.samples = []                # store tuples of (image_path, class_label)
        self.class_to_idx = {}           # map class names to numerical labels
        self.transform = transform       # image transformations for processing and augmentation

        # getting all the class folders from the root directory
        classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

        # if specific classes are provided, only include those classes
        if class_filter:
            classes = [c for c in class_filter if c in classes]

        # assigning each class a unique integer label
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

        # iterating through each class folder and collecting image file paths
        for cls_name in classes:
            cls_path = os.path.join(root_dir, cls_name)
            for fname in os.listdir(cls_path):
                # only include JPEG image files
                if fname.lower().endswith((".jpg")):
                    # storing image path with its corresponding class label
                    self.samples.append((os.path.join(cls_path, fname), self.class_to_idx[cls_name]))

    # returning total number of samples in the dataset
    def __len__(self):
        return len(self.samples)

    # loading an image
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]                            # getting the image path and corresponding numerical label
        class_name = os.path.basename(os.path.dirname(img_path))       # getting the class name from the parent directory
        image = Image.open(img_path).convert("RGB")                    # loading the image and convertign to RGB format

        # applying preprocessing transforms
        if self.transform:
            image = self.transform(image)

        return image, label, class_name

# wrapper dataset used to apply tranforms after splitting the dataset
class TransformDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset            # dataset subset (train/validation/test)
        self.transform = transform      # transformations

    # returning total number of samples in the dataset
    def __len__(self):
        return len(self.subset)

    # loading an image
    def __getitem__(self, idx):
        image, label, class_name = self.subset[idx]                    # getting image, label and class name from subset

        # applying transformations if it has been specified
        if self.transform:
            image = self.transform(image)
        return image, label, class_name

# -- image enhancements -- #
# Contrast Limited Adaptive Histogram Equalisation (CLAHE)
# used to enhance local contrast and improve visibility of retinal features
def apply_clahe(img):
    img = np.array(img)

    # converting RGB image to lab colour space
    # l = lightness, a/b = colour channels
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # applying CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    # merging enhanced lightness channel with original colour channels
    merged = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_Lab2RGB)

    return Image.fromarray(enhanced)

# applying CLAHE randomly during training
# prevents over-reliance on enhanced images
class CLAHETransform:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            return apply_clahe(img)
        return img

# adding gaussian noise to improve robustness
# helps the model generalise to variations on image quality
class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.02):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std + self.mean
        return tensor + noise

# -- dataloaders -- #
# creates batches of data for training, validation and testing
def create_loaders(train_dataset, val_dataset, test_dataset, batch_size=32, pin_memory=True):
    # disable pinned memory if CUDA is unavailable
    if pin_memory and not torch.cuda.is_available():
        pin_memory = False
        print("cuda is not available, setting pin_memory=False")

    # training loader (shuffled for randomness)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=pin_memory
    )

    # validation loader (no shuffling)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=pin_memory
    )

    # test loader (no shuffling)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader

# -- transforms -- #
train_transform = transforms.Compose([
    CLAHETransform(p=0.5),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.2
    ),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2)),
    transforms.ToTensor(),
    AddGaussianNoise(std=0.01),
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
