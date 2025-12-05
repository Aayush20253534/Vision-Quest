import os
import shutil
import random

source_folder = r"C:\Users\LENOVO\Desktop\archive (1)\birds_test" 
destination_folder = r"C:\Users\LENOVO\Desktop\Project\dataset"   
train_ratio = 0.8
image_extensions = (".jpg", ".jpeg", ".png")
single_class_name = "non_drone" 

train_folder = os.path.join(destination_folder, "train", single_class_name)
val_folder = os.path.join(destination_folder, "val", single_class_name)
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)

all_images = [
    f for f in os.listdir(source_folder)
    if f.lower().endswith(image_extensions) and os.path.isfile(os.path.join(source_folder, f))
]

random.shuffle(all_images)

train_count = int(len(all_images) * train_ratio)
train_images = all_images[:train_count]
val_images = all_images[train_count:]

for img in train_images:
    shutil.copy(os.path.join(source_folder, img), os.path.join(train_folder, img))
for img in val_images:
    shutil.copy(os.path.join(source_folder, img), os.path.join(val_folder, img))

print(f"Class: {single_class_name}, Total: {len(all_images)}, Train: {len(train_images)}, Val: {len(val_images)}")
print(f"Images copied to '{destination_folder}'")
