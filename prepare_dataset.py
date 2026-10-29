import os
import shutil
import random

# Source dataset path
source_dir = 'mask_alert_project/dataset'

# Destination paths
dest_images_train = 'data/images/train'
dest_images_val = 'data/images/val'
dest_labels_train = 'data/labels/train'
dest_labels_val = 'data/labels/val'

# Create directories
os.makedirs(dest_images_train, exist_ok=True)
os.makedirs(dest_images_val, exist_ok=True)
os.makedirs(dest_labels_train, exist_ok=True)
os.makedirs(dest_labels_val, exist_ok=True)

# Classes
classes = ['with_mask', 'without_mask']
class_ids = {'with_mask': 0, 'without_mask': 1}

# Collect all images
all_images = []
for cls in classes:
    cls_dir = os.path.join(source_dir, cls)
    if os.path.exists(cls_dir):
        images = [f for f in os.listdir(cls_dir) if f.endswith('.jpg')]
        for img in images:
            all_images.append((os.path.join(cls_dir, img), cls))

# Shuffle
random.shuffle(all_images)

# Split 80% train, 20% val
split_idx = int(0.8 * len(all_images))
train_images = all_images[:split_idx]
val_images = all_images[split_idx:]

# Function to copy and create label
def process_images(image_list, dest_img_dir, dest_label_dir):
    for img_path, cls in image_list:
        # Copy image
        img_name = os.path.basename(img_path)
        shutil.copy(img_path, os.path.join(dest_img_dir, img_name))

        # Create label file
        label_file = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(dest_label_dir, label_file)
        class_id = class_ids[cls]
        # Full image bbox: class x_center y_center width height (normalized)
        with open(label_path, 'w') as f:
            f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")

# Process train and val
process_images(train_images, dest_images_train, dest_labels_train)
process_images(val_images, dest_images_val, dest_labels_val)

print("Dataset preparation complete.")
