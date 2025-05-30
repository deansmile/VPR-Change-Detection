import os
import cv2
import numpy as np

# Define the directories
concat_imgs_dir = "/scratch/ds5725/alvpr/vpr_change_detection_data/3000_q1_q3/cd_data/test/concat_imgs"
predicted_masks_dir = "/scratch/ds5725/alvpr/Robust-Scene-Change-Detection/our_train_visual_2"
output_dir = "/scratch/ds5725/alvpr/Robust-Scene-Change-Detection/output_concatenated_images"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Loop through image pair IDs in the predicted change mask folder
for filename in os.listdir(predicted_masks_dir):
    if filename.endswith(".png"):
        image_pair_id = filename  # Get image pair ID

        # Construct full paths
        concat_img_path = os.path.join(concat_imgs_dir, image_pair_id)
        pred_mask_path = os.path.join(predicted_masks_dir, image_pair_id)
        output_path = os.path.join(output_dir, image_pair_id)

        # Load images
        if not os.path.exists(concat_img_path):
            print(f"Skipping {image_pair_id}, concatenated image not found.")
            continue

        concat_img = cv2.imread(concat_img_path)
        pred_mask = cv2.imread(pred_mask_path)

        # Resize concatenated image to (504, 1512)
        concat_img_resized = cv2.resize(concat_img, (1512, 504), interpolation=cv2.INTER_CUBIC)

        # Concatenate images horizontally without resizing the predicted mask
        final_img = np.hstack((concat_img_resized, pred_mask))

        # Save the output image
        cv2.imwrite(output_path, final_img)

print("Processing complete.")
