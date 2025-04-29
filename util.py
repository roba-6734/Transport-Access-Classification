import os
import shutil
import random
from pathlib import Path


source_dir = 'dataset' 
output_base = 'processed_dataset'  

train_split = 0.7
val_split = 0.15
test_split = 0.15


splits = ['train', 'val', 'test']
class_names = [name for name in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, name))]

for split in splits:
    for class_name in class_names:
        split_path = os.path.join(output_base, split, class_name)
        os.makedirs(split_path, exist_ok=True)


for class_name in class_names:
    class_path = os.path.join(source_dir, class_name)
    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(images)

    total = len(images)
    train_end = int(total * train_split)
    val_end = train_end + int(total * val_split)

    split_data = {
        'train': images[:train_end],
        'val': images[train_end:val_end],
        'test': images[val_end:]
    }

    for split in splits:
        for img in split_data[split]:
            src = os.path.join(class_path, img)
            dst = os.path.join(output_base, split, class_name, img)
            shutil.copy2(src, dst)

print(" Dataset split completed into train, val, and test folders.")
