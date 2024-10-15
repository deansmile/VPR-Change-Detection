import cv2
import torch
import numpy as np
import os
import supervision as sv
from supervision.draw.color import ColorPalette
# from utils.supervision_utils import CUSTOM_COLOR_MAP
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from tqdm import tqdm


CUSTOM_COLOR_MAP = [
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#0082c8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#d2f53c",
    "#fabebe",
    "#008080",
    "#e6beff",
    "#aa6e28",
    "#fffac8",
    "#800000",
    "#aaffc3",
]


# environment settings
# use bfloat16
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# build SAM2 image predictor
sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
sam2_predictor = SAM2ImagePredictor(sam2_model)

# build grounding dino from huggingface
model_id = "./IDEA-Research-grounding-dino-tiny/"
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(model_id)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

source_txt_file = './text.txt'  # Replace with your actual text file path

# Initialize a list to store all the lists
all_lists = []
current_list = []

# Open the source text file and read the lines
with open(source_txt_file, 'r') as file:
    for line in file:
        # Strip any leading/trailing whitespace
        stripped_line = line.strip()
        
        # Check if the line starts with a numbered list item (e.g., "1.", "2.", etc.)
        if stripped_line and stripped_line[0].isdigit() and stripped_line[1] == '.':
            # If a new "1." is encountered and there is a current list, store the current list
            if stripped_line.startswith("1.") and current_list:
                all_lists.append(current_list)
                current_list = []
            
            # Append the numbered item to the current list
            modified_line = stripped_line.split(' ', 1)[1].lower().rstrip('.') + '.'
            
            # Append the modified line to the current list
            current_list.append(modified_line)
    
    # After the loop, append the last list if it exists
    if current_list:
        all_lists.append(current_list)

image_paths=[]
with open("./img_pair.txt", "r") as file:
    for line in file:
        # Split the line into three strings
        parts = line.strip().split()
        if len(parts) >= 2:
            # Get the image paths (first and second strings)
            image_paths.append(parts[0])
            image_paths.append(parts[1])
        if len(image_paths)>=len(all_lists):
            break
print(len(image_paths))
for i in tqdm(range(len(image_paths))):        
    # if i!=5:
    #     continue
    # img_path = '/scratch/ds5725/LineFinder/YOLOv8-HumanDetection/queue.jpg'
    img_path=image_paths[i]
    image = Image.open(img_path)

    sam2_predictor.set_image(np.array(image.convert("RGB")))

    # setup the input image and text prompt for SAM 2 and Grounding DINO
    # VERY important: text queries need to be lowercased + end with a dot
    # texts = ["people in the line."]
    texts=all_lists[i]
    # texts=['green awning.', 'bicycle lane markings.']
    input_boxes = []
    class_names = []  # List to store all class names
    confidences = []  # List to store all confidences

    for text in texts:
        inputs = processor(images=image, text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = grounding_model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.4,
            text_threshold=0.3,
            target_sizes=[image.size[::-1]]
        )

        # Check if any boxes are detected
        if len(results[0]["boxes"]) == 0:
            print(f"No boxes detected for text: '{text}'")
            continue  # Skip if no boxes are detected

        # Append boxes, labels, and confidences
        input_boxes.append(results[0]["boxes"])
        class_names.extend(results[0]["labels"])  # Extend the list of class names
        confidences.extend(results[0]["scores"].cpu().numpy().tolist())  # Extend the list of scores

    # Ensure there are detected boxes
    if len(input_boxes) == 0:
        print(img_path)
        continue
        # raise ValueError("No valid detections found for any text prompt.")

    # Concatenate all boxes into a single array
    input_boxes = torch.cat(input_boxes, dim=0).cpu().numpy()

    # Debugging: Print lengths of the lists to ensure they match
    print(f"Number of detected boxes: {len(input_boxes)}")
    print(f"Number of class names: {len(class_names)}")
    print(f"Number of confidences: {len(confidences)}")

    # Create class IDs, assuming unique IDs for each label
    unique_labels = list(set(class_names))
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    class_ids = np.array([label_to_id[label] for label in class_names])  # Convert labels to numerical IDs

    # Validate that class_ids length matches input_boxes
    if class_ids.shape[0] != input_boxes.shape[0]:
        raise ValueError("Mismatch between the number of class IDs and detected boxes.")

    # Get masks, scores, and logits using the SAM 2 predictor
    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    # Convert the shape to (n, H, W)
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    # Prepare labels for visualization
    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence
        in zip(class_names, confidences)
    ]

    # Visualize image with supervision useful API
    img = cv2.imread(img_path)
    detections = sv.Detections(
        xyxy=input_boxes,  # (n, 4)
        mask=masks.astype(bool),  # (n, h, w)
        class_id=class_ids
    )


    """
    Note that if you want to use default color map,
    you can set color=ColorPalette.DEFAULT
    """
    box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

    label_annotator = sv.LabelAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    # cv2.imwrite("056_1_02_0_box.jpg", annotated_frame)

    mask_annotator = sv.MaskAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    cv2.imwrite("./test_res/"+str(i)+"_blm.jpg", annotated_frame)

    mask_annotator = sv.MaskAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
    annotated_frame = mask_annotator.annotate(scene=img.copy(), detections=detections)
    cv2.imwrite("./test_res/"+str(i)+"_mask.jpg", annotated_frame)
    path = os.path.join("./test_res/"+str(i)+"_mask.jpg")
    print(path)


# 0 if all masks are incorrect, 
# 1 if all masks are new objects, 
# 2 if all masks are of existing object appearance change 
# 3 if all masks are new object due to viewpoint change