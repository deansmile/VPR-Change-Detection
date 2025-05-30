import sys
import os

# Add the parent directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), "grounding_dino"))

from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2

model = load_model("/scratch/ds5725/alvpr/Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py", "/scratch/ds5725/alvpr/GroundingDINO/weights/groundingdino_swint_ogc.pth")
IMAGE_PATH = "/scratch/ds5725/alvpr/cd_datasets/VL-CMU-CD-binary255/test/t0/007_1_00_0.png"
TEXT_PROMPT = "bench."
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.35

print("Image path: "+IMAGE_PATH)
print("Text prompt: "+TEXT_PROMPT)
image_source, image = load_image(IMAGE_PATH)

boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD
)

annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
cv2.imwrite("gdino_007_1_00_0.jpg", annotated_frame)