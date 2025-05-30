import os
import shutil

# Define file paths
input_file = "/scratch/ds5725/alvpr/vpr_change_detection_data/3000_q1_q3/test_1000.txt"
source_dir = "/scratch/ds5725/alvpr/Robust-Scene-Change-Detection/res_comp_imgs"
dest_dir = "/scratch/ds5725/alvpr/Robust-Scene-Change-Detection/test_res_comp_imgs"

# Ensure the destination directory exists
os.makedirs(dest_dir, exist_ok=True)

# Extract test image pair IDs
test_image_ids = set()

with open(input_file, "r") as file:
    for line in file:
        parts = line.strip().split(" ")
        if len(parts) != 2:
            continue  # Skip malformed lines

        image_pair_id, image_path = parts
        if "/test/t0/" in image_path:
            test_image_ids.add(image_pair_id)

# Move test images from source to destination
for image_id in test_image_ids:
    src_path = os.path.join(source_dir, f"{image_id}.png")
    dest_path = os.path.join(dest_dir, f"{image_id}.png")

    if os.path.exists(src_path):
        shutil.move(src_path, dest_path)
    else:
        print(f"Skipping {image_id}.png, file not found.")

print("Moving process completed.")
