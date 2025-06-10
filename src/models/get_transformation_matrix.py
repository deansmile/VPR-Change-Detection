import numpy as np
import cv2

data = np.load('matched_keypoints.npz')
pts1 = data['keypoints0']
pts2 = data['keypoints1']

# print(pts1,pts2)
# Compute homography matrix with RANSAC
H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)  # Warp img2 to img1
print(H)
np.save('warp_matrix.npy', H)
