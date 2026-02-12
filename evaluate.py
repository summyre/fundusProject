import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import FundusDataset
from transforms import val_transform
from train_baseline import simpleCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = ["Healthy", "Diabetic Retinopathy", "Central Serous Chorioretinopathy", "Disc Edema", "Glaucoma", "Macular Scar", "Myopia", "Retinal Detachment", "Retinitis Pigmentosa"]

dataset = FundusDataset(
    root_dir=r"data/Augmented_Dataset",
    transform=val_transform,
    class_filter=class_names
)

# load same val split
val_indices = torch.load("val_indices.pt")
val_dataset = Subset(dataset, val_indices)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# -- model -- #
model = simpleCNN(num_classes=9).to(device)
model.load_state_dict(torch.load("baseline_cnn.pth", map_location=device))
model.eval()

# -- evaluation -- #
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels, _ in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = outputs.argmax(1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# metrics
print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds, target_names=class_names, digits=3))

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(10,8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion Matrix - Baseline CNN (9 classes)")
plt.tight_layout()
plt.show()