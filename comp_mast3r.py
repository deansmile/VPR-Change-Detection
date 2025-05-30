import os
import cv2
import numpy as np

# Define directories
test_imgs_dir = "/scratch/ds5725/alvpr/vpr_change_detection_data/3000_q1_q3/cd_data/test_orig/concat_imgs"
train_imgs_dir="/scratch/ds5725/alvpr/vpr_change_detection_data/3000_q1_q3/cd_data/train/concat_imgs"
rscd_masks_dir = "/scratch/ds5725/alvpr/Robust-Scene-Change-Detection/our_train_visual_1000"
mast3r_masks_dir = "/scratch/ds5725/alvpr/mast3r/test_1000_sam_gpt_filter"
output_dir = "/scratch/ds5725/alvpr/Robust-Scene-Change-Detection/res_comp_imgs"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Process each image pair ID
for filename in os.listdir(rscd_masks_dir):
    if filename.endswith(".png"):
        image_pair_id = filename  # Get image pair ID

        train_imgs_path=os.path.join(train_imgs_dir, image_pair_id)
        test_imgs_path=os.path.join(test_imgs_dir, image_pair_id)
        if os.path.exists(train_imgs_path):
            concat_img_path=train_imgs_path
        else:
            concat_img_path=test_imgs_path
        # Construct full paths
        # concat_img_path = os.path.join(concat_imgs_dir, image_pair_id)
        rscd_mask_path = os.path.join(rscd_masks_dir, image_pair_id)
        mast3r_mask_path = os.path.join(mast3r_masks_dir, "res_"+image_pair_id)
        output_path = os.path.join(output_dir, image_pair_id)

        # Load images
        if not os.path.exists(concat_img_path) or not os.path.exists(mast3r_mask_path):
            print(f"Skipping {image_pair_id}, missing required images.")
            continue

        concat_img = cv2.imread(concat_img_path)
        rscd_mask = cv2.imread(rscd_mask_path)
        mast3r_mask = cv2.imread(mast3r_mask_path)

        # Extract t0 and t1 images (first 1280 pixels width)
        image_pair = concat_img[:, :1280, :]

        # Add white space padding to maintain width 1920
        padding = np.full((480, 640, 3), 255, dtype=np.uint8)  # White padding
        image_pair_padded = np.hstack((image_pair, padding))

        # Convert non-black pixels in ground truth mask and MAST3R mask to red
        # mast3r_mask[np.any(mast3r_mask != [0, 0, 0], axis=-1)] = [0, 0, 255]
        ground_truth_mask = concat_img[:, 1280:, :]
        # ground_truth_mask[np.any(ground_truth_mask != [0, 0, 0], axis=-1)] = [0, 0, 255]

        # Resize RSCD mask to (480, 640)
        rscd_mask_resized = cv2.resize(rscd_mask, (640, 480), interpolation=cv2.INTER_CUBIC)

        # Concatenate ground truth mask, MAST3R mask, and RSCD mask horizontally
        combined_masks = np.hstack((ground_truth_mask, mast3r_mask, rscd_mask_resized))

        # Concatenate image pair with three masks vertically
        final_img = np.vstack((image_pair_padded, combined_masks))

        # Save the output image
        cv2.imwrite(output_path, final_img)

print("Processing complete.")
