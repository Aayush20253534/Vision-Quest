import os
import shutil
import random

source_folder = r"C:\Users\LENOVO\Desktop\file"        
destination_folder = r"C:\Users\LENOVO\Desktop\Project\dataset"  # destination
train_ratio = 0.8
image_extensions = (".jpg", ".jpeg", ".png")
single_class_name = "drone" 


subfolders = [f for f in os.listdir(source_folder) if os.path.isdir(os.path.join(source_folder, f))]

if not subfolders: 
    subfolders = [single_class_name]
    source_subfolders = {single_class_name: source_folder}
else:  
    source_subfolders = {f: os.path.join(source_folder, f) for f in subfolders}

for class_name in subfolders:
    class_path = source_subfolders[class_name]

    train_folder = os.path.join(destination_folder, "train", class_name)
    val_folder = os.path.join(destination_folder, "val", class_name)
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)

    all_images = [
        f for f in os.listdir(class_path)
        if f.lower().endswith(image_extensions) and os.path.isfile(os.path.join(class_path, f))
    ]

    random.shuffle(all_images)
    train_count = int(len(all_images) * train_ratio)
    train_images = all_images[:train_count]
    val_images = all_images[train_count:]

    for img in train_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(train_folder, img))
    for img in val_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(val_folder, img))

    print(f"Class: {class_name}, Total: {len(all_images)}, Train: {len(train_images)}, Val: {len(val_images)}")

print(f"All images copied to '{destination_folder}'")
