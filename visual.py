import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the original image (BGR)
img = cv2.imread("12315_t0.png")

# Load the mask image (BGR)
mask = cv2.imread("12315.png")

# Convert mask to RGB for easier color filtering
mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

# Create a binary mask for red areas (R=255, G=0, B=0)
red_mask = np.all(mask_rgb == [255, 0, 0], axis=-1).astype(np.uint8)  # 1 where red, 0 elsewhere

# Create a red overlay image of same size
overlay = np.zeros_like(img)
overlay[:, :] = [0, 0, 255]  # Red in BGR

# Blend overlay with original image where red_mask == 1
alpha = 0.5  # transparency
img_with_overlay = img.copy()
img_with_overlay[red_mask == 1] = cv2.addWeighted(img[red_mask == 1], 1 - alpha, overlay[red_mask == 1], alpha, 0)

# Convert to RGB for matplotlib display
img_rgb = cv2.cvtColor(img_with_overlay, cv2.COLOR_BGR2RGB)

# Show the result
plt.imshow(img_rgb)
# plt.title("Red Mask Overlay")
plt.axis("off")
plt.show()
