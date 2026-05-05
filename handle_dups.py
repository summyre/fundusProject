# this code handles cross class duplicates
import os
import shutil
import hashlib
from collections import defaultdict

# setting the names of the directories
dataset_dir = "data/Original_Dataset"
conflict_dir = "data/conflicts/og"

# making conflict directory if it does not exist
os.makedirs(conflict_dir, exist_ok=True)

# generating a unique MD5 hash for each image file
def compute_hash(image_path):
    hasher = hashlib.md5()
    with open(image_path, "rb") as f:    # read image file in binary mode
        buf = f.read()
        hasher.update(buf)               # update hash object with file contents
    return hasher.hexdigest()            # return final hash value as a hexadecimal string

# grouping images by hash value - any images with the same hash are considered duplicates
hash_map = defaultdict(list)

# iterating through each disease class folder
for class_name in os.listdir(dataset_dir):
    class_path = os.path.join(dataset_dir, class_name)
    if not os.path.isdir(class_path):    # skip non-directory files
        continue

    # processing each image within the class folder
    for img in os.listdir(class_path):
        img_path = os.path.join(class_path, img)
        if not img.lower().endswith(".jpg"):    # only process JPEGs
            continue

        img_hash = compute_hash(img_path)                    # compute image hash
        hash_map[img_hash].append((class_name, img_path))    # store class label and file path under the hash

conflict_count = 0

# checking for duplicate images
for img_hash, entries in hash_map.items():
    classes = set(c for c, _ in entries)

    if len(classes) > 1:
        conflict_count += 1
        # moving conflicting images to separate folder
        for class_name, img_path in entries:
            new_name = f"{class_name}_{os.path.basename(img_path)}"
            new_path = os.path.join(conflict_dir, new_name)
            shutil.move(img_path, new_path)

print(f"moved {conflict_count} conflicting image groups to '{conflict_dir}'")
