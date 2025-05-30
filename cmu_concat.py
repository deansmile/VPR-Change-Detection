import os
from PIL import Image
from tqdm import tqdm

# Define the input directories
folder_t0 = "/scratch/ds5725/alvpr/cd_datasets/PSCD/crop_test/t0"
folder_t1 = "/scratch/ds5725/alvpr/cd_datasets/PSCD/crop_test/t1"
folder_mask = "/scratch/ds5725/alvpr/cd_datasets/PSCD/crop_test/mask"
folder_pred = "/scratch/ds5725/alvpr/Robust-Scene-Change-Detection/pretrain_pscd_visual"

# Optional: output directory to save concatenated images
output_dir = "/scratch/ds5725/alvpr/Robust-Scene-Change-Detection/pretrain_pscd_visual_concat"
os.makedirs(output_dir, exist_ok=True)

# Get the list of image names (assuming all folders have the same files)
image_names = sorted(os.listdir(folder_t0))

# Process each image
for name in tqdm(image_names):
    path_t0 = os.path.join(folder_t0, name)
    path_t1 = os.path.join(folder_t1, name)
    path_mask = os.path.join(folder_mask, name)
    path_pred = os.path.join(folder_pred, name)

    # Load images
    img_t0 = Image.open(path_t0)
    img_t1 = Image.open(path_t1)
    img_mask = Image.open(path_mask)
    img_pred = Image.open(path_pred)

    # Concatenate images horizontally
    width, height = img_t0.size
    total_width = width * 4
    new_img = Image.new("RGB", (total_width, height))
    new_img.paste(img_t0, (0, 0))
    new_img.paste(img_t1, (width, 0))
    new_img.paste(img_mask.convert("RGB"), (2 * width, 0))  # convert mask if needed
    new_img.paste(img_pred.convert("RGB"), (3 * width, 0))

    # Save result
    new_img.save(os.path.join(output_dir, name))
