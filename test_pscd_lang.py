import copy
import glob
import os
import pickle
import numpy as np
from PIL import Image

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

changed_objects = parse_cmu_changed_objects("/scratch/ds5725/alvpr/gpt/text_pscd.txt")
print(changed_objects["00000623.png"])
# i=0
# for key, value in changed_objects.items():
#     print(f"{key}: {value}")
#     i+=1
#     if i>10:
#         break

# with open("changed_object_pscd.pkl", "wb") as f:
#     pickle.dump(changed_object_pscd, f)