import numpy as np
import torch
import torchvision.transforms.functional as tvff
from torch.nn import functional as F
from torchvision import transforms as tvf
from PIL import Image
import evaluation
import os
import pickle
import cv2
# local module
from py_utils.src import utils, utils_torch, utils_img


def translate_image(tx, ty):

    def f(img):
        orig_shape = img.shape
        if len(orig_shape) == 2:
            # affine function cannot accept 2D image or an error will be raised
            # RuntimeError: grid_sampler(): expected grid ...
            # thus, change shape from (m, n) to (1, m, n)
            img = img.unsqueeze(0)

        x = tvf.functional.affine(
            img, angle=0, translate=(tx, ty), scale=1.0, shear=0
        )

        if len(orig_shape) == 2:
            x = x.squeeze(0)

        return x

    return f


def rotate_image(angle):

    def f(img):
        orig_shape = img.shape
        if len(orig_shape) == 2:
            # affine function cannot accept 2D image or an error will be raised
            # RuntimeError: grid_sampler(): expected grid ...
            # thus, change shape from (m, n) to (1, m, n)
            img = img.unsqueeze(0)

        x = tvf.functional.affine(
            img, angle=angle, translate=(0, 0), scale=1.0, shear=0
        )

        if len(orig_shape) == 2:
            x = x.squeeze(0)

        return x

    return f


class CDDataWrapper:

    def __init__(
        self,
        dataset,
        transform=None,
        target_transform=None,
        return_ind=False,
        translate0=(0, 0),
        translate1=(0, 0),
        rotate_angle0=0.0,
        rotate_angle1=0.0,
        hflip_prob=0.0,
        augment_diff_degree=None,
        augment_diff_translate=None,
    ):

        if transform is None:

            def transform(x):
                return x

        if target_transform is None:

            def target_transform(x):
                return x

        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform
        self.return_ind = return_ind
        self.hflip_prob = hflip_prob

        self.translate0 = translate_image(*translate0)
        self.translate1 = translate_image(*translate1)
        self.rotate0 = rotate_image(rotate_angle0)
        self.rotate1 = rotate_image(rotate_angle1)
        self._pre_transform = tvf.ToTensor()
        self._pos_transform = tvf.ToPILImage()

        if augment_diff_degree is None:
            augment_diff_degree = 0.0

        self.augment_diff_degree = np.abs(augment_diff_degree)
        self.augment_diff_translate = augment_diff_translate

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        t0, t1, gt, caption = self.dataset[idx]

        t0 = self._pre_transform(t0)
        t1 = self._pre_transform(t1)
        gt = self._pre_transform(gt)

        t0 = self.translate0(t0)
        t1 = self.translate1(t1)
        gt = self.translate0(gt)

        t0 = self.rotate0(t0)
        t1 = self.rotate1(t1)
        gt = self.rotate0(gt)

        t0 = self._pos_transform(t0)
        t1 = self._pos_transform(t1)
        gt = self._pos_transform(gt)

        t0 = self.transform(t0)
        t1 = self.transform(t1)
        gt = self.target_transform(gt)

        if self.augment_diff_degree > 0.0:

            degree = np.random.uniform(
                -self.augment_diff_degree,
                self.augment_diff_degree,
            )

            t0 = rotate_image(degree)(t0)
            gt = rotate_image(degree)(gt)

        if self.augment_diff_translate is not None:

            translate = np.random.uniform(
                self.augment_diff_translate[0],
                self.augment_diff_translate[1],
                size=2,
            )

            t0 = translate_image(*translate)(t0)
            gt = translate_image(*translate)(gt)

        if np.random.random() < self.hflip_prob:
            t0 = tvff.hflip(t0)
            t1 = tvff.hflip(t1)
            gt = tvff.hflip(gt)

        output = t0, t1, gt, caption

        if self.return_ind:
            return idx, output

        return output


def _yield_CD_evaluation_after_cropping(model, dataset, crop_shape):

    device = utils_torch.get_model_device(model)

    for input_1, input_2, target in dataset:

        input_1 = input_1.to(device)
        input_2 = input_2.to(device)

        with torch.no_grad():
            predict = model(input_1, input_2)  # (batch, m, n, 2)
            predict = torch.argmax(predict, dim=-1)  # (batch, m, n)

        target = target.detach().cpu().numpy()  # (batch, m, n)
        predict = predict.detach().cpu().numpy()  # (batch, m, n)

        assert len(target.shape) == 3 and len(predict.shape) == 3
        assert target.shape == predict.shape

        # ravel out each item in a batch
        for pre, tar in zip(predict, target):

            tar = utils_img.center_crop_image(tar, crop_shape)
            pre = utils_img.center_crop_image(pre, crop_shape)

            R = evaluation.change_mask_metric(pre, tar)
            yield R["precision"], R["recall"], R["accuracy"], R["f1_score"]

def extract_image_ids(folder, suffix=".png"):
    return sorted({os.path.splitext(f)[0] for f in os.listdir(folder) if f.endswith(suffix)}, key=int)

def extract_ids_from_pkl(folder):
    ids = set()
    for fname in os.listdir(folder):
        if fname.endswith(".pkl"):
            parts = os.path.splitext(fname)[0].split("_")
            if parts[-1].isdigit():
                ids.add(parts[-1])
    return ids


def load_masks(pkl_folder, prefix, image_id):
    pkl_path = os.path.join(pkl_folder, f"{prefix}_{image_id}.pkl")
    # print(pkl_path,os.path.exists(pkl_path))
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    return []

def mask_overlap_ratio(mask, predict):
    resized = cv2.resize(mask.astype(np.uint8), (predict.shape[1], predict.shape[0]), interpolation=cv2.INTER_NEAREST)
    intersection = np.logical_and(resized, predict).sum()
    mask_area = resized.sum()
    return intersection / mask_area if mask_area > 0 else 0, resized

def concat_with_margins(image_path, predict_before, predict_after, target_mask, target_size=(224, 224), margin_width=10):
    """
    Concatenates original image + three masks with blue margins in between.
    All masks must be 2D binary masks (0 or 1).
    Output is a BGR image.
    """
    assert predict_before.shape == predict_after.shape == target_mask.shape, "All masks must be the same shape"

    h, w = target_size

    # Load and resize the original image
    original = cv2.imread(image_path)
    if original is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    original_resized = cv2.resize(original, (w, h), interpolation=cv2.INTER_AREA)
    if len(original_resized.shape) == 2:  # grayscale
        original_resized = cv2.cvtColor(original_resized, cv2.COLOR_GRAY2BGR)

    # Convert binary masks to 3-channel BGR
    def to_bgr(mask):
        gray = (mask * 255).astype(np.uint8)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    pb_color = to_bgr(predict_before)
    pa_color = to_bgr(predict_after)
    tgt_color = to_bgr(target_mask)

    # Create blue vertical margin
    blue_margin = np.full((h, margin_width, 3), (255, 0, 0), dtype=np.uint8)

    # print(original_resized.shape,pb_color.shape,pa_color.shape,tgt_color.shape)
    # Concatenate: image | margin | before | margin | after | margin | target
    result = np.hstack((
        original_resized,
        blue_margin,
        pb_color,
        blue_margin,
        pa_color,
        blue_margin,
        tgt_color
    ))

    return result


def _yield_CD_evaluation_sam(model, dataset):
    # print("yield_CD_evaluation")
    t0_folder = "/scratch/ds5725/alvpr/vpr_change_detection_data/3000_q1_q3/cd_data/test/t0"
    pkl_folders_main = {
        "objects": "/scratch/ds5725/alvpr/Grounded-SAM-2/test_1000_objects_new",
        "plants":  "/scratch/ds5725/alvpr/Grounded-SAM-2/test_1000_plants"
    }
    pkl_folders_rest = {
        "objects": "/scratch/ds5725/alvpr/Grounded-SAM-2/test_1000_rest_objects",
        "plants":  "/scratch/ds5725/alvpr/Grounded-SAM-2/test_1000_rest_plants"
    }
    t0_ids = extract_image_ids(t0_folder)
    pkl_ids = extract_ids_from_pkl(pkl_folders_main["objects"]).union(extract_ids_from_pkl(pkl_folders_main["plants"]))
    rest_ids = extract_ids_from_pkl(pkl_folders_rest["objects"]).union(extract_ids_from_pkl(pkl_folders_rest["plants"]))
    
    i=0
    device = utils_torch.get_model_device(model)
    
    for input_1, input_2, target in dataset:

        image_id=t0_ids[i]
        gsam_flag=1
        if image_id in pkl_ids:
            folders = pkl_folders_main
        elif image_id in rest_ids:
            folders = pkl_folders_rest
        else:
            gsam_flag=0

        input_1 = input_1.to(device)
        input_2 = input_2.to(device)

        with torch.no_grad():
            predict = model(input_1, input_2)  # (batch, m, n, 2)
            predict = torch.argmax(predict, dim=-1)  # (batch, m, n)

        target = target.detach().cpu().numpy()  # (batch, m, n)
        predict = predict.detach().cpu().numpy()  # (batch, m, n)
        print("predict shape",predict.shape)
        assert len(target.shape) == 3 and len(predict.shape) == 3
        assert target.shape == predict.shape
        
        # ravel out each item in a batch
        for pre, tar in zip(predict, target):
            pre_before=pre
            pre=np.zeros(pre.shape, dtype=np.uint8)
            if gsam_flag==1:
                for category, folder in folders.items():
                    masks = load_masks(folder, f"all_plants", image_id)

                    for mask in masks:
                        if not isinstance(mask, np.ndarray):
                            continue  # Skip invalid data
                        ratio, resized_mask = mask_overlap_ratio(mask, pre)
                        # if ratio < 0.3:
                        pre = np.logical_or(pre, resized_mask).astype(np.uint8)

            R = evaluation.change_mask_metric(pre, tar)
            yield R["precision"], R["recall"], R["accuracy"], R["f1_score"]
        concatenated = concat_with_margins(t0_folder+"/"+str(image_id)+".png",pre_before, pre, tar)
        cv2.imwrite("/scratch/ds5725/alvpr/Robust-Scene-Change-Detection/sam_predict_concat_test/comp_"+str(image_id)+".png",concatenated)
        i+=1

def _yield_CD_evaluation_pscd(model, dataset):

    t0_folder = "/scratch/ds5725/alvpr/cd_datasets/PSCD/crop_test/t0"
    pkl_folder = "/scratch/ds5725/alvpr/Grounded-SAM-2/pscd_objects"
    t0_ids = sorted(os.listdir(t0_folder))

    device = utils_torch.get_model_device(model)

    i=0
    for input_1, input_2, target in dataset:
        name = dataset.dataset.dataset.filenames[i]
        image_id = name.replace(".png", "")
        pkl_path = os.path.join(pkl_folder, f"{image_id}.pkl")

        input_1 = input_1.to(device)
        input_2 = input_2.to(device)

        with torch.no_grad():
            predict = model(input_1, input_2)  # (batch, m, n, 2)
            predict = torch.argmax(predict, dim=-1)  # (batch, m, n)

        target = target.detach().cpu().numpy()  # (batch, m, n)
        predict = predict.detach().cpu().numpy()  # (batch, m, n)

        assert len(target.shape) == 3 and len(predict.shape) == 3
        assert target.shape == predict.shape

        # ravel out each item in a batch
        for pre, tar in zip(predict, target):
            pre_before=pre
            pre=np.zeros(pre.shape, dtype=np.uint8)
            if os.path.exists(pkl_path):
                with open(pkl_path, "rb") as f:
                    masks = pickle.load(f)
                    for mask in masks:
                        if isinstance(mask, np.ndarray):
                            resized = cv2.resize(mask.astype(np.uint8), pre.shape, interpolation=cv2.INTER_NEAREST)
                            pre = np.logical_or(pre, resized).astype(np.uint8)
            R = evaluation.change_mask_metric(pre, tar)
            yield R["precision"], R["recall"], R["accuracy"], R["f1_score"]
            concatenated = concat_with_margins(t0_folder+"/"+str(image_id)+".png",pre_before, pre, tar)
            cv2.imwrite("/scratch/ds5725/alvpr/Robust-Scene-Change-Detection/pscd_concat/"+str(image_id)+".png",concatenated)
        i+=1

def _yield_CD_evaluation_cmu(model, dataset):

    t0_folder = "/scratch/ds5725/alvpr/cd_datasets/VL-CMU-CD-binary255/test/t0"
    pkl_folder = "/scratch/ds5725/alvpr/Grounded-SAM-2/cmu_objects"
    t0_ids = sorted(os.listdir(t0_folder))

    device = utils_torch.get_model_device(model)

    i=0
    for input_1, input_2, target in dataset:

        image_id=os.path.splitext(t0_ids[i])[0]
        pkl_path = os.path.join(pkl_folder, f"{image_id}.pkl")

        input_1 = input_1.to(device)
        input_2 = input_2.to(device)

        with torch.no_grad():
            predict = model(input_1, input_2)  # (batch, m, n, 2)
            predict = torch.argmax(predict, dim=-1)  # (batch, m, n)

        target = target.detach().cpu().numpy()  # (batch, m, n)
        predict = predict.detach().cpu().numpy()  # (batch, m, n)

        assert len(target.shape) == 3 and len(predict.shape) == 3
        assert target.shape == predict.shape

        # ravel out each item in a batch
        for pre, tar in zip(predict, target):
            pre_before=pre
            pre=np.zeros(pre.shape, dtype=np.uint8)
            if os.path.exists(pkl_path):
                with open(pkl_path, "rb") as f:
                    masks = pickle.load(f)
                    for mask in masks:
                        if isinstance(mask, np.ndarray):
                            resized = cv2.resize(mask.astype(np.uint8), pre.shape, interpolation=cv2.INTER_NEAREST)
                            pre = np.logical_or(pre, resized).astype(np.uint8)
            R = evaluation.change_mask_metric(pre, tar)
            yield R["precision"], R["recall"], R["accuracy"], R["f1_score"]
            concatenated = concat_with_margins(t0_folder+"/"+str(image_id)+".png",pre_before, pre, tar)
            cv2.imwrite("/scratch/ds5725/alvpr/Robust-Scene-Change-Detection/cmu_concat/"+str(image_id)+".png",concatenated)
        i+=1

def process_object_desc(object_desc):
    PREPOSITIONS = ["on", "in", "at", "near", "without"]
    # Convert to lowercase
    object_desc = object_desc.lower()
    # Remove quotes
    object_desc = object_desc.replace('"', '').replace('“', '').replace('”', '')

    # Remove prepositions and everything after them
    words = object_desc.split()
    for i, word in enumerate(words):
        if word in PREPOSITIONS:
            object_desc = " ".join(words[:i])  # Keep only words before the preposition
            break

    # Ensure it ends with a period if it's not empty
    object_desc = object_desc.strip()
    if object_desc and not object_desc.endswith('.'):
        object_desc += '.'

    if "none" in object_desc:
        return None
    return object_desc
    
def parse_cmu_changed_objects(file_path):
    cmu_objects = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()

    current_image_id = None
    current_objects = []

    for line in lines:
        line = line.strip()

        # Detect image pair ID line
        if line.endswith('.png'):
            # Save the previous entry
            if current_image_id is not None:
                cmu_objects[current_image_id] = current_objects

            # Use the first image name as key
            current_image_id = os.path.basename(line.split()[0])
            current_objects = []

        # Capture numbered lines
        elif line.lstrip().startswith(tuple([f"{i}." for i in range(1, 100)])):
            object_desc = line.split('.', 1)[1].strip()
            processed_desc = process_object_desc(object_desc)
            if processed_desc:
                current_objects.append(processed_desc)

    # Save last entry
    if current_image_id is not None:
        cmu_objects[current_image_id] = current_objects

    return cmu_objects

def read_image_ids_and_paths(file_path):
    image_ids = []
    image_paths = []

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                image_ids.append(parts[0])
                image_paths.append(parts[1])

    return image_ids, image_paths

def _yield_CD_evaluation(model, dataset):

    # changed_objects = parse_cmu_changed_objects('/scratch/ds5725/alvpr/gpt/text_cmu.txt')
    # folder_t0 = "/scratch/ds5725/alvpr/cd_datasets/VL-CMU-CD-binary255/test/t0"
    # image_names = sorted(os.listdir(folder_t0))

    # image_paths=[]
    # image_ids=[]
    # for name in image_names:
    #     path_t0 = os.path.join(folder_t0, name)
    #     image_paths.append(path_t0)
    #     image_ids.append(name)

    # for id in image_ids[:5]:
    #     print(id,changed_objects[id])
    # exit()
    device = utils_torch.get_model_device(model)

    for input_1, input_2, target, caption in dataset:
        input_1 = input_1.to(device)
        input_2 = input_2.to(device)
        # print(input_1.shape, input_2.shape)
        # print(f"Caption passed to improve_dino_feature: {caption}")
        with torch.no_grad():
            predict = model(input_1, input_2, caption)  # (batch, m, n, 2)
            predict = torch.argmax(predict, dim=-1)  # (batch, m, n)
        # print("predict shape",predict.shape)

        target = target.detach().cpu().numpy()  # (batch, m, n)
        predict = predict.detach().cpu().numpy()  # (batch, m, n)

        assert len(target.shape) == 3 and len(predict.shape) == 3
        assert target.shape == predict.shape

        # ravel out each item in a batch
        for pre, tar in zip(predict, target):

            R = evaluation.change_mask_metric(pre, tar)
            yield R["precision"], R["recall"], R["accuracy"], R["f1_score"]

        # exit()


def image_change_detection_evaluation(
    model,
    data_loader,
    evaluator=None,
    prefix="",
    device="cuda",
    verbose=True,
    dry_run=False,
    return_details=False,
    return_duration=False,
):
    """
    Evaluate the performance of a change detection model on a given dataset.

    Args:
        model (torch.nn.Module):
            The change detection model to be evaluated.
        data_loader (torch.utils.data.DataLoader):
            DataLoader for the evaluation dataset.
        evaluator (generator, optional):
            Custom evaluator yielding evaluation metrics.
            If None, a generator is formed by both `model` and `data_loader`.
            If not None, `model` and `data_loader` are ignored.
        prefix (str, optional):
            Prefix for progress messages. Default is an empty string.
        device (str, optional):
            The device to run the evaluation on ('cuda' or 'cpu').
            Default is 'cuda'.
        verbose (bool, optional):
            If True, print progress information during evaluation.
            Default is True.
        dry_run (bool, optional):
            If True, perform only one iteration for testing purposes.
            Default is False.
        return_details (bool, optional):
            If True, include detailed metrics for each iteration in the output.
            Default is False.
        return_duration (bool, optional):
            If True, include the total duration of the evaluation in the output.
            Default is False.

    Returns:
        tuple: A tuple containing:
            - dict: Aggregate evaluation metrics with keys
                'precision', 'recall', 'accuracy', and 'f1_score'.
            - float (optional): Total duration of the evaluation in seconds,
                if return_duration is True.
            - dict (optional): Detailed metrics for each iteration,
                if return_details is True.
    """
    verbose=False
    
    precisions = []
    recalls = []
    accuracies = []
    f1_scores = []

    model.eval()
    model = model.to(device)

    if evaluator is None:
        evaluator = _yield_CD_evaluation(model, data_loader)

    progress = utils.ProgressTimer(verbose=verbose, prefix=prefix)
    progress.tic(total_items=len(data_loader.dataset))

    for output in evaluator:

        prec, rec, acc, f1 = output
        precisions.append(prec)
        recalls.append(rec)
        accuracies.append(acc)
        f1_scores.append(f1)

        progress.toc(add=1)

        if dry_run:
            break

    R = {
        "precision": np.mean(precisions),
        "recall": np.mean(recalls),
        "accuracy": np.mean(accuracies),
        "f1_score": np.mean(f1_scores),
    }

    details = {
        "precision": precisions,
        "recall": recalls,
        "accuracy": accuracies,
        "f1_score": f1_scores,
    }

    output = (R,)

    if return_duration:
        output += (progress.total_seconds,)

    if return_details:
        output += (details,)

    return output
