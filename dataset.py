from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os

class FundusDataset(Dataset):
    def __init__(self, root_dir, transform=None, class_filter=None):
        self.samples = []
        self.class_to_idx = {}
        self.transform = transform

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

        if self.transform:
            image = self.transform(image)

        return image, label, class_name
