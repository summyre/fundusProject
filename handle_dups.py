# this code handles cross class duplicates
import os
import shutil
import hashlib
from collections import defaultdict

dataset_dir = "data/Augmented_Dataset"
conflict_dir = "data/conflicts/aug"

os.makedirs(conflict_dir, exist_ok=True)

def compute_hash(image_path):
    hasher = hashlib.md5()
    with open(image_path, "rb") as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

hash_map = defaultdict(list)

for class_name in os.listdir(dataset_dir):
    class_path = os.path.join(dataset_dir, class_name)
    if not os.path.isdir(class_path):
        continue
    
    for img in os.listdir(class_path):
        img_path = os.path.join(class_path, img)
        if not img.lower().endswith(".jpg"):
            continue

        img_hash = compute_hash(img_path)
        hash_map[img_hash].append((class_name, img_path))

conflict_count = 0

for img_hash, entries in hash_map.items():
    classes = set(c for c, _ in entries)

    if len(classes) > 1:
        conflict_count += 1
        for class_name, img_path in entries:
            new_name = f"{class_name}_{os.path.basename(img_path)}"
            new_path = os.path.join(conflict_dir, new_name)
            shutil.move(img_path, new_path)

print(f"moved {conflict_count} conflicting image groups to '{conflict_dir}'")