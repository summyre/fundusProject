import os
import hashlib
from collections import Counter
import matplotlib.pyplot as plt
from PIL import Image
import torch

data_root = r"data/Augmented_Dataset"

def image_hash(image_path):
    hasher = hashlib.md5()
    with open(image_path, "rb") as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

class_counts = {}
image_paths = []

for class_name in os.listdir(data_root):
    class_path = os.path.join(data_root, class_name)
    if not os.path.isdir(class_path):
        continue
    
    images = [
        os.path.join(class_path, f)
        for f in os.listdir(class_path)
        if f.lower().endswith(".jpg")
    ]
    
    class_counts[class_name] = len(images)
    image_paths.extend(images)

print("Number of images per class:")
for cls, count in class_counts.items():
    print(f"{cls}: {count}")


# -- plot class distribution -- #
plt.figure(figsize=(10, 5))
plt.bar(class_counts.keys(), class_counts.values())
plt.xticks(rotation=45, ha="right")
plt.ylabel("Number of images")
plt.title("Class distribution in augmented dataset")
plt.tight_layout()
plt.show()

"""
# -- duplicate detection -- #
hashes = {}
duplicates = []

for img_path in image_paths:
    h = image_hash(img_path)
    if h in hashes:
        duplicates.append((hashes[h], img_path))
    else:
        hashes[h] = img_path

print("\nDuplicate image check:")
if len(duplicates) == 0:
    print("no duplicate images found")
else:
    print(f"found {len(duplicates)} duplicate pairs:")
    for original, duplicate in duplicates[:5]:
        print(f"duplicate:\n {original}\n {duplicate}")
"""
duplicates = [1,2]

# -- summary -- #
total_images = sum(class_counts.values())
print(f"""
------ Dataset Summary ------
total images: {total_images}
number of classes: {len(class_counts)}
duplicate images: {len(duplicates)}
""")

# -- torch check -- #
a = torch.cuda.is_available()
id = torch.cuda.current_device()
name = torch.cuda.get_device_name(id)
memory = torch.cuda.device(id)
gpu = torch.cuda.device_count()
print(f"\navailable: {a}\ncurrent device: {id} | {name}\ngpu: {memory} | {gpu}")