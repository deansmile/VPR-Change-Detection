import sys
import os
import cv2
import torch
import numpy as np
from pathlib import Path
import argparse

# Ensure SuperGlue model can be imported
sys.path.append('/scratch/zl4701/CYWS-3D/SuperGluePretrainedNetwork')
from models.matching import Matching

# --- Utility: Draw and save matches ---
def draw_and_save_matches(img0, img1, mkpts0, mkpts1, save_path='match_output.png'):
    if len(img0.shape) == 2:
        img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
    if len(img1.shape) == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)

    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    out_img = np.zeros((max(h0, h1), w0 + w1, 3), dtype=np.uint8)
    out_img[:h0, :w0] = img0
    out_img[:h1, w0:] = img1

    for pt0, pt1 in zip(mkpts0, mkpts1):
        pt0 = tuple(np.round(pt0).astype(int))
        pt1 = tuple(np.round(pt1).astype(int) + np.array([w0, 0]))
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.line(out_img, pt0, pt1, color=color, thickness=1)
        cv2.circle(out_img, pt0, 3, color, -1)
        cv2.circle(out_img, pt1, 3, color, -1)

    cv2.imwrite(save_path, out_img)
    print(f"[✓] Match visualization saved to: {save_path}")


# --- Argument Parser ---
parser = argparse.ArgumentParser()
parser.add_argument('--img0', type=str, default='/scratch/zl4701/CYWS-3D/demo_data/24478_831_2_0.png')
parser.add_argument('--img1', type=str, default='/scratch/zl4701/CYWS-3D/demo_data/24478_831_2_1.png')
parser.add_argument('--out', type=str, default='matched_keypoints.npz')
parser.add_argument('--vis', type=str, default='match_output.png')
args = parser.parse_args()

# --- SuperGlue Configuration ---
superglue_config = {
    'superpoint': {
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': 1024,
    },
    'superglue': {
        'weights': 'indoor',
        'sinkhorn_iterations': 20,
        'match_threshold': 0.2,
    }
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[INFO] Using device: {device}")

# --- Initialize Model ---
matching = Matching(superglue_config).eval().to(device)

# --- Load Images ---
def read_image_gray(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return image

def resize_pad(image, target_size=(640, 480)):
    h, w = image.shape
    scale = min(target_size[0] / h, target_size[1] / w)
    resized = cv2.resize(image, (int(scale * w), int(scale * h)))
    padded = np.zeros(target_size, dtype=np.uint8)
    padded[:resized.shape[0], :resized.shape[1]] = resized
    return padded

image0 = resize_pad(read_image_gray(args.img0))
image1 = resize_pad(read_image_gray(args.img1))

# --- Preprocess ---
image0_tensor = torch.from_numpy(image0 / 255.).float()[None, None].to(device)
image1_tensor = torch.from_numpy(image1 / 255.).float()[None, None].to(device)

data = {'image0': image0_tensor, 'image1': image1_tensor}

# --- Inference ---
with torch.no_grad():
    pred = matching(data)

keypoints0 = pred['keypoints0'][0].cpu().numpy()
keypoints1 = pred['keypoints1'][0].cpu().numpy()
matches = pred['matches0'][0].cpu().numpy()

# --- Extract Matched Keypoints ---
valid = matches > -1
matched_kpts0 = keypoints0[valid]
matched_kpts1 = keypoints1[matches[valid]]

# --- Save Matched Keypoints ---
np.savez(args.out, keypoints0=matched_kpts0, keypoints1=matched_kpts1)
print(f"[✓] Saved {matched_kpts0.shape[0]} matched keypoints to '{args.out}'.")

# --- Draw and Save ---
draw_and_save_matches(image0, image1, matched_kpts0, matched_kpts1, args.vis)
