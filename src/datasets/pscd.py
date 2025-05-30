import glob
import os

import numpy as np
from PIL import Image

_file_path = os.path.split(os.path.realpath(__file__))[0]

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

class PSCD:

    def __init__(self, root, mode="train"):

        assert mode in {"train", "test", "val"}

        self.mode = mode
        self.root = root

        # for simplicity, this class do not perform any security check
        # the assumptions are:
        # 1. [t0, t1, mask_t0, mask_t1] folders must contain in root
        # 2. all files in [t0, t1, mask_t0, mask_t1] use the exactly
        #    same file name.
        self._t0 = os.path.join(self.root, "t0")
        self._t1 = os.path.join(self.root, "t1")
        self._mask_t0 = os.path.join(self.root, "mask_t0")
        self._mask_t1 = os.path.join(self.root, "mask_t1")

        filenames = glob.glob(os.path.join(self._t0, "*.png"))
        filenames = [os.path.split(i)[-1] for i in filenames]
        filenames = sorted(filenames)
        filenames = np.array(filenames)

        path = os.path.join(_file_path, f"indices/pscd.{mode}.index")
        with open(path, "r") as fd:
            indices = fd.read().splitlines()

        I = np.searchsorted(filenames, indices)
        assert np.all(filenames[I] == np.array(indices))

        self._filenames = filenames[I]

    @property
    def filenames(self):
        return self._filenames

    def __len__(self):
        return len(self._filenames)

    def __getitem__(self, idx):

        filename = self._filenames[idx]
        t0 = os.path.join(self._t0, filename)
        t1 = os.path.join(self._t1, filename)
        t0_mask = os.path.join(self._mask_t0, filename)
        t1_mask = os.path.join(self._mask_t1, filename)

        t0_image = Image.open(t0).convert("RGB")
        t1_image = Image.open(t1).convert("RGB")
        t0_mask = Image.open(t0_mask)
        t1_mask = Image.open(t1_mask)

        t0_image = np.array(t0_image) / 255.0
        t1_image = np.array(t1_image) / 255.0
        t0_mask = np.array(t0_mask) / 255.0
        t1_mask = np.array(t1_mask) / 255.0

        t0_mask = t0_mask > 0.0
        t1_mask = t1_mask > 0.0

        t0_image = t0_image.astype(np.float32)
        t1_image = t1_image.astype(np.float32)
        t0_mask = t0_mask.astype(np.float32)
        t1_mask = t1_mask.astype(np.float32)

        return t0_image, t1_image, t0_mask, t1_mask

    @property
    def figsize(self):
        return np.array([224, 1024])


class CroppedPSCD(PSCD):
    def __init__(
        self,
        root,
        mode="train",
        crop_num=8,  # 4096 / 512 = 8 horizontal crops
        use_mask_t0=True,
        use_mask_t1=False,
    ):
        if crop_num <= 1:
            raise ValueError("crop_num must be greater than 1")

        super().__init__(root, mode)

        self.filename_to_caption = parse_cmu_changed_objects('/scratch/ds5725/alvpr/gpt/text_pscd.txt')

        self.crop_width = 512
        self.crop_height = 512
        self.crop_num = crop_num
        self.use_mask_t0 = use_mask_t0
        self.use_mask_t1 = use_mask_t1

        self._crop_filenames = None

        # Calculate horizontal stride (no overlap)
        self.stride = (4096 - self.crop_width) // (self.crop_num - 1)

    @property
    def original_filenames(self):
        return np.repeat(self._filenames, self.crop_num)

    @property
    def filenames(self):
        if self._crop_filenames is not None:
            return self._crop_filenames

        N = len(self._filenames)
        names = self.original_filenames
        indices = np.tile(np.arange(self.crop_num), N)

        filenames = [
            name.replace(".png", f".{ind}.png")
            for name, ind in zip(names, indices)
        ]
        self._crop_filenames = np.array(filenames)
        return self._crop_filenames

    def __len__(self):
        return len(self._filenames) * self.crop_num

    def __getitem__(self, idx):
        id_image = idx // self.crop_num
        id_crop = idx % self.crop_num

        # Get the original filename like '00000000.png'
        filename = self._filenames[id_image]
        # filename = os.path.splitext(filename)[0]  # '00000000'
        if filename not in self.filename_to_caption:
            print(f"Warning: No caption found for {filename}")
        caption = self.filename_to_caption.get(filename, "")

        t0_image, t1_image, t0_mask, t1_mask = super().__getitem__(id_image)

        # Center crop vertically to 512px
        top = (t0_image.shape[0] - self.crop_height) // 2
        bottom = top + self.crop_height

        # Horizontal crop based on index
        left = self.stride * id_crop
        right = left + self.crop_width

        t0_image = t0_image[top:bottom, left:right, :]
        t1_image = t1_image[top:bottom, left:right, :]
        t0_mask = t0_mask[top:bottom, left:right]
        t1_mask = t1_mask[top:bottom, left:right]

        output = (t0_image, t1_image)
        if self.use_mask_t0:
            output += (t0_mask,)
        if self.use_mask_t1:
            output += (t1_mask,)
        output += (caption,)
        return output

    @property
    def figsize(self):
        return np.array([512, 512])


class DiffViewPSCD(CroppedPSCD):

    def __init__(
        self,
        root,
        mode="train",
        crop_num=15,
        adjacent_distance=1,
    ):

        super().__init__(
            root,
            mode=mode,
            crop_num=crop_num,
            use_mask_t0=True,
            use_mask_t1=False,
        )

        if adjacent_distance * self.stride >= 224:
            x = 224 // self.stride
            msg = f"adjacent_distance must be within [{-x}, {x})"
            raise ValueError(msg)

        # indices for every t0
        start = 0 if adjacent_distance >= 0 else -adjacent_distance
        end = 0 if adjacent_distance < 0 else adjacent_distance

        start_ind = np.arange(len(self._filenames)) * crop_num

        indices = np.arange(start, crop_num - end)
        indices = indices[None, :] + start_ind[:, None]
        indices = indices.flatten()

        self.adjacent_distance = adjacent_distance
        self.indices = indices

        self._diff_crop_filenames = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        idx = self.indices[idx]
        t0_image, _, t0_mask = super().__getitem__(idx)
        _, t1_image, _ = super().__getitem__(idx + self.adjacent_distance)

        if self.adjacent_distance >= 0:
            t0_mask[:, : self.adjacent_distance * self.stride] = 0.0
        else:
            t0_mask[:, self.adjacent_distance * self.stride :] = 0.0

        return t0_image, t1_image, t0_mask

    @property
    def filenames(self):

        if self._diff_crop_filenames is not None:
            return self._diff_crop_filenames

        names = super().filenames

        t0_names = names[self.indices]
        t1_names = names[self.indices + self.adjacent_distance]

        names = []
        for t0_name, t1_name in zip(t0_names, t1_names):
            t0_name = t0_name.replace(".png", "")
            t1_name = t1_name.replace(".png", "")

            name, t0_ind = t0_name.split(".")
            _, t1_ind = t1_name.split(".")
            name = f"{name}.{t0_ind}-{t1_ind}.png"
            names.append(name)
        self._diff_crop_filenames = np.array(names)
        return self._diff_crop_filenames
