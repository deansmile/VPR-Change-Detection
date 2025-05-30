import os

# Folder paths
t0_folder = "/scratch/ds5725/alvpr/vpr_change_detection_data/3000_q1_q3/cd_data/test/t0"

pkl_folders = [
    "/scratch/ds5725/alvpr/Grounded-SAM-2/test_1000_objects_new",
    "/scratch/ds5725/alvpr/Grounded-SAM-2/test_1000_plants"
]

rest_pkl_folders = [
    "/scratch/ds5725/alvpr/Grounded-SAM-2/test_1000_rest_objects",
    "/scratch/ds5725/alvpr/Grounded-SAM-2/test_1000_rest_plants"
]

# Extract image IDs from t0 (strip .png)
def extract_t0_ids(folder):
    return sorted({os.path.splitext(f)[0] for f in os.listdir(folder) if f.endswith(".png")}, key=int)

# Extract image IDs from pkl filenames like all_plants_123.pkl
def extract_ids_from_pkl(folder):
    ids = set()
    for f in os.listdir(folder):
        if f.endswith(".pkl"):
            parts = os.path.splitext(f)[0].split("_")
            if parts[-1].isdigit():
                ids.add(parts[-1])
    return ids

# Collect IDs
t0_ids = extract_t0_ids(t0_folder)

pkl_ids = set()
for folder in pkl_folders:
    pkl_ids.update(extract_ids_from_pkl(folder))

rest_union_ids = set()
for folder in rest_pkl_folders:
    rest_union_ids.update(extract_ids_from_pkl(folder))

# Create mapping
coverage_dict = {}
for image_id in t0_ids:
    if image_id in pkl_ids:
        coverage_dict[image_id] = 1
    elif image_id in rest_union_ids:
        coverage_dict[image_id] = 2
    else:
        coverage_dict[image_id] = 0

# Output summary
print(f"Total t0 image IDs: {len(t0_ids)}")
print(f"Covered by pkl_ids (1): {sum(1 for v in coverage_dict.values() if v == 1)}")
print(f"Covered by rest_union_ids (2): {sum(1 for v in coverage_dict.values() if v == 2)}")
print(f"Uncovered (0): {sum(1 for v in coverage_dict.values() if v == 0)}")

# Example output
print("\nExample mapping (first 10 IDs):")
for k in list(coverage_dict.keys())[:10]:
    print(f"{k}: {coverage_dict[k]}")

