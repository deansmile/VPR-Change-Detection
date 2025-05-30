import copy
import glob
import os

import numpy as np
from PIL import Image

_file_path = os.path.split(os.path.realpath(__file__))[0]

import random
from torch.utils.data import Dataset
import pickle


def process_object_desc(object_desc):
    PREPOSITIONS = ["on", "in", "at", "near", "without"]
    # Convert to lowercase
    object_desc = object_desc.lower()
    # Remove quotes
    object_desc = object_desc.replace('"', '').replace('“', '').replace('”', '')

    # Remove prepositions and everything after them
    words = object_desc.split()
    # for i, word in enumerate(words):
    #     if word in PREPOSITIONS:
    #         object_desc = " ".join(words[:i])  # Keep only words before the preposition
    #         break

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
                cmu_objects[current_image_id] = " ".join(current_objects)

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
        cmu_objects[current_image_id] = " ".join(current_objects)

    return cmu_objects
    
# Function to parse text_1000_veg.txt and strictly keep the first list
def parse_veg_file(file_path):
    veg_objects = {}
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    current_pair_id = None
    current_objects = []
    first_list_captured = False  # Ensure only the first list is stored

    for line in lines:
        line = line.strip()
        
        # Check for a new image pair
        if line.endswith('.png'):
            # If there is a previous pair, save its results
            if current_pair_id is not None:
                veg_objects[current_pair_id] = " ".join(current_objects)

            # Extract the image_pair_id from the first filename
            current_pair_id = os.path.basename(line.split()[0])
            current_objects = []
            first_list_captured = False  # Reset rule to allow capturing a new first list

        # Start capturing when encountering a numbered list (1., 2., ..., 99.)
        elif line.lstrip().startswith(tuple([f"{i}." for i in range(1, 100)])):  
            if not first_list_captured:  # Only capture the first list found
                object_desc = line.split('.', 1)[1].strip()
                processed_desc = process_object_desc(object_desc)
                if processed_desc:
                    current_objects.append(processed_desc)

        # Stop capturing after the first list has been processed
        elif current_objects:  # If we already captured at least one item, stop further capturing
            first_list_captured = True  # Mark that the first list is stored

    # Save the last image pair
    if current_pair_id is not None:
        veg_objects[current_pair_id] = " ".join(current_objects)

    return veg_objects

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

def get_original_name(filename):
    if filename.endswith('_rot.png'):
        return filename.replace('_rot.png', '.png')
    return filename

class Our_dataset(Dataset):
    """
    Dataset for change detection with a new naming convention.
    Each image pair has an ID, e.g., `1.png` in `t0`, `t1`, and `mask` folders.
    """

    def __init__(self, root, mode="train", val_ratio=0.1, seed=42):
        """
        Args:
            root (str): Root directory containing train and test folders.
            mode (str): One of {"train", "test", "val"}.
            val_ratio (float): Percentage of training data used for validation.
            seed (int): Random seed for reproducibility.
        """
        
        mode = mode.lower()
        assert mode in {"train", "test", "val"}, "Mode must be one of 'train', 'test', or 'val'."
        
        if mode in {"train", "val"}:
            postfix = "train"
            print("initialize our train dataset")
        else:
            postfix = "test"
            print("initialize our test dataset")
        
        # print("change object text description from GPT")
        # changed_plant_text = parse_veg_file('/scratch/ds5725/alvpr/gpt/text_1000_rest_veg.txt')
        # changed_plant_text1 = parse_veg_file('/scratch/ds5725/alvpr/gpt/text_1000_veg.txt')
        # changed_plant_text_combined = changed_plant_text | changed_plant_text1

        # # print("changed object text description")
        # changed_object_text = parse_veg_file('/scratch/ds5725/alvpr/gpt/text_1000_rest.txt')
        # changed_object_text1 = parse_veg_file('/scratch/ds5725/alvpr/gpt/text_1000.txt')
        # changed_object_text_combined = changed_object_text | changed_object_text1
        # self.filename_to_caption = {k: changed_object_text_combined[k] + " " + changed_plant_text_combined[k] for k in changed_object_text_combined.keys()}

        print("change object text description from InternVL")
        self.filename_to_caption = parse_veg_file('/scratch/ds5725/alvpr/gpt/text_internvl.txt')

        for i, (k, v) in enumerate(self.filename_to_caption.items()):
            if i >= 5:
                break
            print(f"{k}: {v}")

        self.mode = mode
        self.root = os.path.join(root, postfix)
        self.t0_path = os.path.join(self.root, "t0")
        self.t1_path = os.path.join(self.root, "t1")
        self.mask_path = os.path.join(self.root, "mask")
        
        # Get all filenames (just the numeric IDs without extensions)
        filenames = glob.glob(os.path.join(self.t0_path, "*.png"))
        filenames = [os.path.split(i)[-1] for i in filenames]
        filenames = sorted(filenames)  
        
        if mode in {"train", "val"}:
            random.shuffle(filenames)  # Shuffle before splitting
            val_count = int(len(filenames) * val_ratio)
            
            if mode == "train":
                filenames = filenames[val_count:]
            else:  # mode == "val"
                filenames = filenames[:val_count]
        
        self._filenames = np.array(filenames)
    
    def __len__(self):
        return len(self._filenames)
    
    def __getitem__(self, idx):
        t0_image = self.get_img_0(idx)
        t1_image = self.get_img_1(idx)
        mask_image = self.get_mask(idx)
        filename = get_original_name(self._filenames[idx])
        
        if filename not in self.filename_to_caption:
            print(f"Warning: No caption found for {filename}")
        caption = self.filename_to_caption.get(filename, "")
        # print(filename,caption)
        return t0_image, t1_image, mask_image, caption
    
    def get_img_0(self, idx):
        filename = self._filenames[idx]
        path = os.path.join(self.t0_path, filename)
        image = Image.open(path).convert("RGB")
        image = image.resize((512, 512))
        image = np.array(image) / 255.0
        image = image.astype(np.float32)
        return image
    
    def get_img_1(self, idx):
        filename = self._filenames[idx]
        path = os.path.join(self.t1_path, filename)
        image = Image.open(path).convert("RGB")
        image = image.resize((512, 512))
        image = np.array(image) / 255.0
        image = image.astype(np.float32)
        return image
    
    def get_mask(self, idx):
        filename = self._filenames[idx]
        mask_path = os.path.join(self.mask_path, filename)
        mask_image = Image.open(mask_path).convert("L")
        mask_image = mask_image.resize((512, 512))
        mask_image = np.array(mask_image) / 255.0
        mask_image = mask_image > 0.0  # Convert to binary mask
        mask_image = mask_image.astype(np.float32)
        return mask_image
    
    @property
    def filenames(self):
        return self._filenames
    
    @property
    def figsize(self):
        return np.array([512, 512])

class S2LookingDataset(Dataset):
    """
    Dataset for the S2Looking change detection task.
    Uses Image1 as t0, Image2 as t1, and mask for ground truth.
    """

    def __init__(self, root, mode="train"):
        """
        Args:
            root (str): Root directory of the S2Looking dataset.
            mode (str): One of "train", "val", "test".
        """
        mode = mode.lower()
        assert mode in {"train", "val", "test"}, "Mode must be 'train', 'val', or 'test'"
        
        with open("/scratch/ds5725/alvpr/SegEarth-OV/changed_object_text.pkl", "rb") as f:
            self.filename_to_caption = pickle.load(f)
        self.mode = mode
        self.root = os.path.join(root, mode)
        self.t0_path = os.path.join(self.root, "Image1")
        self.t1_path = os.path.join(self.root, "Image2")
        self.mask_path = os.path.join(self.root, "mask")

        # List all filenames
        filenames = glob.glob(os.path.join(self.t0_path, "*.png"))
        self._filenames = sorted([os.path.basename(f) for f in filenames])

    def __len__(self):
        return len(self._filenames)

    def __getitem__(self, idx):
        filename = self._filenames[idx][:-4]  # Remove .png extension
        t0 = self.get_img_0(idx)
        t1 = self.get_img_1(idx)
        mask = self.get_mask(idx)
        if filename.endswith("_1"):
            if filename[:-2] not in self.filename_to_caption:
                print(f"Warning: No caption found for {filename}")
                caption = ""
            else:
                caption_list=self.filename_to_caption[filename[:-2]]["B"]
                caption = " ".join(caption_list)
        else:
            if filename not in self.filename_to_caption:
                print(f"Warning: No caption found for {filename}")
                caption = ""
            else:
                caption_list = self.filename_to_caption[filename]["A"]
                caption = " ".join(caption_list)
        if caption and not caption.endswith('.'):
            caption += '.'
        # if filename not in self.filename_to_caption:
        #     print(f"Warning: No caption found for {filename}")
        return t0, t1, mask, caption

    def get_img_0(self, idx):
        filename = self._filenames[idx]
        path = os.path.join(self.t0_path, filename)
        image = Image.open(path).convert("RGB").resize((512, 512))
        image = np.array(image).astype(np.float32) / 255.0
        return image

    def get_img_1(self, idx):
        filename = self._filenames[idx]
        path = os.path.join(self.t1_path, filename)
        image = Image.open(path).convert("RGB").resize((512, 512))
        image = np.array(image).astype(np.float32) / 255.0
        return image

    def get_mask(self, idx):
        filename = self._filenames[idx]
        path = os.path.join(self.mask_path, filename)
        image = Image.open(path).convert("L").resize((512, 512))
        mask = np.array(image).astype(np.float32) / 255.0
        return (mask > 0).astype(np.float32)

    @property
    def filenames(self):
        return self._filenames

    @property
    def figsize(self):
        return np.array([512, 512])

class VL_CMU_CD:
    """
    each image should be <group_id>_1_<seq_id>_<angle>.png
    """

    def __init__(self, root, mode="train"):
        
        print("initialize cmu dataset")
        
        mode = mode.lower()
        assert mode in {"train", "test", "val"}
        if mode in {"train", "val"}:
            postfix = "train"
        else:
            postfix = "test"

        self.filename_to_caption = parse_cmu_changed_objects('/scratch/ds5725/alvpr/gpt/text_cmu_'+postfix+'.txt')
        # for key, val in list(self.filename_to_caption.items())[:5]:
        #     print(f"{key}: {val}")

        self.mode = mode

        # for simplicity, this class do not perform any security check
        # the assumptions are:
        # 1. [t0, t1, mask] folder must contain in root / train|test
        # 2. all files in [t0, t1, mask] use the exactly same file name.
        self.root = os.path.join(root, postfix)
        self.t0_path = os.path.join(self.root, "t0")
        self.t1_path = os.path.join(self.root, "t1")
        self.mask_path = os.path.join(self.root, "mask")

        filenames = glob.glob(os.path.join(self.t0_path, "*.png"))
        filenames = [os.path.split(i)[-1] for i in filenames]
        filenames = sorted(filenames)

        if mode in {"train", "val"}:

            path = os.path.join(_file_path, f"indices/vl-cmu-cd.{mode}.index")
            with open(path, "r") as fd:
                indices = fd.read().splitlines()
            filenames = [i for i in filenames if i.split("_")[0] in indices]

        self._filenames = np.array(filenames)

    def __len__(self):
        return len(self._filenames)

    def __getitem__(self, idx):
        # designed for torch.utils.data.DataLoader, supporting fetching
        # a data sample for a given key.
        # https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset

        t0_image = self.get_img_0(idx)
        t1_image = self.get_img_1(idx)
        mask_image = self.get_mask(idx)
        filename = self._filenames[idx]
        group_id = filename.split("_")[0]
        reference_filename = f"{group_id}_1_00_0.png"
        if reference_filename not in self.filename_to_caption:
            print(f"Warning: No caption found for {filename}")
        caption = self.filename_to_caption.get(reference_filename, "")
        return t0_image, t1_image, mask_image, caption

    def loc(self, key):
        if (
            not isinstance(key, slice)
            and np.shape(key) == ()
            and not (isinstance(key, np.ndarray) and key.dtype == np.bool_)
        ):
            key = np.unique(key)

        other = copy.copy(self)
        other._filenames = self._filenames[key]
        return other

    def get_img_0(self, idx):
        filename = self._filenames[idx]
        path = os.path.join(self.t0_path, filename)
        image = Image.open(path).convert("RGB")
        image = np.array(image) / 255.0
        image = image.astype(np.float32)
        return image

    def get_img_1(self, idx):
        filename = self._filenames[idx]
        path = os.path.join(self.t1_path, filename)
        image = Image.open(path).convert("RGB")
        image = np.array(image) / 255.0
        image = image.astype(np.float32)
        return image

    def get_mask(self, idx):
        filename = self._filenames[idx]
        mask_path = os.path.join(self.mask_path, filename)
        mask_image = Image.open(mask_path)
        mask_image = np.array(mask_image) / 255.0
        mask_image = mask_image > 0.0
        mask_image = mask_image.astype(np.float32)
        return mask_image

    @property
    def filenames(self):
        return self._filenames

    @property
    def group_ids(self):
        return np.array([i.split("_")[0] for i in self._filenames])

    @property
    def angles(self):
        x = [i.split(".")[0] for i in self._filenames]
        x = [int(i.split("_")[-1]) for i in x]
        return np.array(x)

    @property
    def seq_ids(self):
        x = [i.split(".")[0] for i in self._filenames]
        x = [i.split("_")[-2] for i in x]
        return np.array(x)

    @property
    def figsize(self):
        return np.array([512, 512])

class Diff_VL_CMU_CD:

    def __init__(self, root, mode="train", adjacent_distance=1):

        t0_dataset = VL_CMU_CD(root, mode=mode)
        t1_dataset = VL_CMU_CD(root, mode=mode)

        assert adjacent_distance != 0

        if mode in ["train", "val"]:
            stride = 4
        else:
            stride = 1

        if adjacent_distance > 0:
            I = slice(None, -adjacent_distance * stride)
            J = slice(adjacent_distance * stride, None)
        else:
            I = slice(-adjacent_distance * stride, None)
            J = slice(None, adjacent_distance * stride)

        t0_dataset = t0_dataset.loc(I)
        t1_dataset = t1_dataset.loc(J)

        I = t0_dataset.group_ids == t1_dataset.group_ids

        t0_dataset = t0_dataset.loc(I)
        t1_dataset = t1_dataset.loc(I)

        assert np.all(t0_dataset.group_ids == t1_dataset.group_ids)
        assert np.all(t0_dataset.angles == t1_dataset.angles)

        self.t0_dataset = t0_dataset
        self.t1_dataset = t1_dataset

        self._filenames = None

    def __len__(self):
        return len(self.t0_dataset)

    def __getitem__(self, idx):
        t0_image = self.t0_dataset.get_img_0(idx)
        t1_image = self.t1_dataset.get_img_1(idx)
        mask_image = self.t0_dataset.get_mask(idx)

        return t0_image, t1_image, mask_image

    def loc(self, key):
        other = copy.copy(self)
        other.t0_dataset = self.t0_dataset.loc(key)
        other.t1_dataset = self.t1_dataset.loc(key)
        return other

    def get_img_0(self, idx):
        return self.t0_dataset.get_img_0(idx)

    def get_img_1(self, idx):
        return self.t1_dataset.get_img_1(idx)

    def get_mask(self, idx):
        return self.t0_dataset.get_mask(idx)

    @property
    def filenames(self):

        if self._filenames is not None:
            return self._filenames

        names = zip(
            self.t0_dataset.group_ids,
            self.t0_dataset.seq_ids,
            self.t1_dataset.seq_ids,
            self.t0_dataset.angles,
        )

        names = [f"{i}_1_{j}.{k}_{l}.png" for i, j, k, l in names]
        self._filenames = np.array(names)
        return self._filenames

    @property
    def group_ids(self):
        return self.t0_dataset.group_ids

    @property
    def angles(self):
        return self.t0_dataset.angles

    @property
    def seq_ids(self):

        ids = zip(
            self.t0_dataset.seq_ids,
            self.t1_dataset.seq_ids,
        )

        ids = [f"{i}.{j}" for i, j in ids]
        return np.array(ids)

    @property
    def figsize(self):
        return np.array([512, 512])