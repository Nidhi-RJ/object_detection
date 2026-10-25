# run the ann_to_yolo_dataset.py file first before this file.
import random, re
from pathlib import Path
import shutil

val_ratio = 0.3
train_ratio = 0.7

#change the head folder
DATASET_HEAD_FOLDER = Path("datasets/global_daily_object_yolo/Dollar street trial")

# def rename_files(class_folder: Path) -> Path:

#     return class_folder

classid = {}
objects = 0

train_images = DATASET_HEAD_FOLDER / "images/train"
train_labels = DATASET_HEAD_FOLDER / "labels/train"
val_images = DATASET_HEAD_FOLDER / "images/val"
val_labels = DATASET_HEAD_FOLDER / "labels/val"

for folder in (train_images, train_labels, val_images, val_labels):
    folder.mkdir(parents=True, exist_ok=True)

# Reproducible shuffle
random.seed(42)

for class_folder in DATASET_HEAD_FOLDER.iterdir():
    if not class_folder.is_dir():
        continue

    if not class_folder.name.startswith("Objects__"):
        continue

    image_folder = class_folder / "img"
    label_folder = class_folder / "ann_yolo"

    if not image_folder.is_dir() or not label_folder.is_dir():
        print(f"Missing images or ann_yolo folder: {class_folder}")
        continue

    print(f"\nOpening: {class_folder}")

    imagefiles = sorted(
        f for f in image_folder.rglob("*")
        if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    # Match image001.jpg to image001.jpg.txt
    label_lookup = {
        f.name: f
        for f in label_folder.rglob("*.txt")
        if f.is_file()
    }

    pairs = []
    for image in imagefiles:
        label = label_lookup.get(image.name + ".txt")

        # Also support image001.txt if labels use that naming convention
        if label is None:
            label = label_lookup.get(image.stem + ".txt")

        if label is not None:
            pairs.append((image, label))
        else:
            print(f"Missing label for: {image.name}")

    random.shuffle(pairs)

    split_index = int(len(pairs) * train_ratio)
    train_pairs = pairs[:split_index]
    val_pairs = pairs[split_index:]

    def copy_pairs(pairs_to_copy, image_dest, label_dest):
        for image, label in pairs_to_copy:
            dest_image = image_dest / image.name
            dest_label = label_dest / label.name

            if dest_image.exists() or dest_label.exists():
                print(f"Already exists, skipping: {image.name}")
                continue

            shutil.copy2(image, dest_image)
            shutil.copy2(label, dest_label)

    copy_pairs(train_pairs, train_images, train_labels)
    copy_pairs(val_pairs, val_images, val_labels)

    print(f"Matched pairs: {len(pairs)}")
    print(f"Training pairs: {len(train_pairs)}")
    print(f"Validation pairs: {len(val_pairs)}")

print("\nDataset split finished.")