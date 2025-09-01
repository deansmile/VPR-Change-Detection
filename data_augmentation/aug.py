import os
import cv2
import glob

def rotate_image_center(img, angle_degrees):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, -angle_degrees, 1.0)  # Clockwise = negative
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return rotated

def augment_change_dataset(base_dir):
    for split in ['test', 'train']:
        t0_dir = os.path.join(base_dir, split, 't0')
        t1_dir = os.path.join(base_dir, split, 't1')
        mask_dir = os.path.join(base_dir, split, 'mask')

        t0_paths = glob.glob(os.path.join(t0_dir, '*.png'))

        for t0_path in t0_paths:
            pair_id = os.path.basename(t0_path).replace('.png', '')
            rot_pair_id = f"{pair_id}_rot"

            t0_rot_path = os.path.join(t0_dir, f"{rot_pair_id}.png")
            t1_rot_path = os.path.join(t1_dir, f"{rot_pair_id}.png")
            mask_rot_path = os.path.join(mask_dir, f"{rot_pair_id}.png")

            # Skip if already augmented
            if os.path.exists(t0_rot_path) and os.path.exists(t1_rot_path) and os.path.exists(mask_rot_path):
                continue

            # Original file paths
            t1_path = os.path.join(t1_dir, f"{pair_id}.png")
            mask_path = os.path.join(mask_dir, f"{pair_id}.png")

            # Load images
            t0_img = cv2.imread(t0_path)
            t1_img = cv2.imread(t1_path)
            mask_img = cv2.imread(mask_path)

            # Rotate t0 and mask
            t0_rot = rotate_image_center(t0_img, 10)
            mask_rot = rotate_image_center(mask_img, 10)

            # Save augmented images
            cv2.imwrite(t0_rot_path, t0_rot)
            cv2.imwrite(t1_rot_path, t1_img)  # t1 unchanged
            cv2.imwrite(mask_rot_path, mask_rot)

            print(f"Augmented and saved: {rot_pair_id}")

# Run the augmentation
augment_change_dataset("/vast/ds5725/cd_datasets/our_qual")
