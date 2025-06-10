import cv2
import torch
import numpy as np

import sys
sys.path.append('/scratch/zl4701/CYWS-3D/SuperGluePretrainedNetwork')
from models.matching import Matching


class SuperGlueWarpEstimator:
    def __init__(self, superglue_config, device='cuda'):
        self.device = device
        self.matching = Matching(superglue_config).eval().to(device)

    def read_and_preprocess(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        assert image is not None, f"Image not found: {image_path}"
        h, w = image.shape
        scale = min(640 / h, 480 / w)
        resized = cv2.resize(image, (int(scale * w), int(scale * h)))
        padded = np.zeros((640, 480), dtype=np.uint8)
        padded[:resized.shape[0], :resized.shape[1]] = resized
        tensor = torch.from_numpy(padded / 255.).float()[None, None].to(self.device)
        return tensor, padded

    def get_warp_matrix(self, img_path0, img_path1):
        img0_tensor, _ = self.read_and_preprocess(img_path0)
        img1_tensor, _ = self.read_and_preprocess(img_path1)
        data = {'image0': img0_tensor, 'image1': img1_tensor}

        with torch.no_grad():
            pred = self.matching(data)

        kpts0 = pred['keypoints0'][0].cpu().numpy()
        kpts1 = pred['keypoints1'][0].cpu().numpy()
        matches = pred['matches0'][0].cpu().numpy()

        valid = matches > -1
        if valid.sum() < 4:
            return torch.eye(3)  # fallback

        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        H, _ = cv2.findHomography(mkpts1, mkpts0, cv2.RANSAC, 5.0)
        return torch.from_numpy(H).float()

    def get_warp_matrix_from_tensor(self, img_tensor0, img_tensor1):
        def to_gray_np(tensor):
            img = tensor.detach().cpu().numpy()
            img = np.mean(img, axis=0)
            img = (img * 255).astype(np.uint8)
            return img

        img0 = to_gray_np(img_tensor0)
        img1 = to_gray_np(img_tensor1)
        img0_tensor = torch.from_numpy(img0 / 255.).float()[None, None].to(self.device)
        img1_tensor = torch.from_numpy(img1 / 255.).float()[None, None].to(self.device)

        with torch.no_grad():
            pred = self.matching({'image0': img0_tensor, 'image1': img1_tensor})

        kpts0 = pred['keypoints0'][0].cpu().numpy()
        kpts1 = pred['keypoints1'][0].cpu().numpy()
        matches = pred['matches0'][0].cpu().numpy()
        valid = matches > -1

        if valid.sum() < 4:
            return torch.eye(3)  # fallback

        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        H, _ = cv2.findHomography(mkpts1, mkpts0, cv2.RANSAC, 5.0)
        return torch.from_numpy(H).float()


if __name__ == '__main__':
    config = {
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
    estimator = SuperGlueWarpEstimator(config, device='cuda' if torch.cuda.is_available() else 'cpu')

    img0 = '/scratch/zl4701/CYWS-3D/demo_data/24478_831_2_0.png'
    img1 = '/scratch/zl4701/CYWS-3D/demo_data/24478_831_2_1.png'
    H = estimator.get_warp_matrix(img0, img1)
    print("Homography matrix:\n", H)
