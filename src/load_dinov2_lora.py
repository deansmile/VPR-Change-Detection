import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Dinov2Model, Dinov2ImageProcessor
from peft import PeftModel
from PIL import Image

# ================================================================
#  Step 1: Re-define the DinoHead class
#  You need the class definition to instantiate the model before
#  loading its state dictionary.
# ================================================================
class DinoHead(nn.Module):
    def __init__(self,
                 in_dim=768, # DINOv2-Base has 768 dim output
                 hid_dim=2048,
                 bottleneck_dim=256,
                 n_prototypes=65_536):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hid_dim, bias=False),
            nn.GELU(),
            nn.Linear(hid_dim, bottleneck_dim, bias=False),
        )
        self.prototypes = nn.Linear(bottleneck_dim, n_prototypes, bias=False)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        return self.prototypes(x)

# ================================================================
#  Step 2: Load the base model, adapters, and head
# ================================================================
def load_explora_model(adapter_path: str, head_path: str):
    """Loads the base DINOv2, applies LoRA adapters, and loads the head."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 2a. Load the original, pre-trained base model ---
    print("Loading base DINOv2 model...")
    base_model = Dinov2Model.from_pretrained("facebook/dinov2-base")
    print(f"Base model type: {type(base_model)}")

    # --- 2b. Load LoRA adapters and apply them to the base model ---
    # `PeftModel.from_pretrained` takes the base model and the path to the adapters
    # and returns a new model with the adapters merged in.
    print(f"Loading LoRA adapters from '{adapter_path}'...")
    explora_backbone = PeftModel.from_pretrained(base_model, adapter_path)
    print(f"PEFT model type: {type(explora_backbone)}")

    # --- 2c. Load the DINO projection head ---
    print(f"Loading DINO head from '{head_path}'...")
    dino_head = DinoHead() # Instantiate the head
    dino_head.load_state_dict(torch.load(head_path, map_location=device))

    # --- 2d. Set models to evaluation mode ---
    explora_backbone.eval()
    dino_head.eval()
    explora_backbone.to(device)
    dino_head.to(device)

    print("\n✓ Models loaded successfully!")
    return explora_backbone, dino_head


# ================================================================
#  Step 3: Run Inference
# ================================================================
if __name__ == "__main__":
    ADAPTER_PATH = "explora_lora_corrected/student_lora"
    HEAD_PATH = "explora_lora_corrected/dino_head.pt"

    # Load the complete model
    backbone, head = load_explora_model(ADAPTER_PATH, HEAD_PATH)

    # Prepare an image for inference
    processor = Dinov2ImageProcessor.from_pretrained("facebook/dinov2-base")
    # Create a dummy image
    dummy_image = Image.new('RGB', (224, 224), color = 'red')

    # Preprocess the image
    inputs = processor(images=dummy_image, return_tensors="pt").to(backbone.device)

    # Run inference
    with torch.no_grad():
        # Get feature embeddings from the ExPLoRA-adapted backbone
        outputs = backbone(**inputs)
        features = outputs.last_hidden_state # Shape: (1, 257, 768)
        cls_feature = features[:, 0]         # Shape: (1, 768)

        # Get logits from the DINO head
        logits = head(cls_feature)           # Shape: (1, 65536)

    print("\n--- Inference Demo ---")
    print(f"Input image shape (post-processing): {inputs.pixel_values.shape}")
    print(f"Backbone CLS feature shape: {cls_feature.shape}")
    print(f"DINO head output logits shape: {logits.shape}")

    # ================================================================
    #  Optional but Recommended for Inference: Merge LoRA weights
    # ================================================================
    # For maximum inference speed, you can merge the LoRA weights directly
    # into the base model. This eliminates the PEFT overhead entirely.
    # The model becomes a standard `Dinov2Model` again.
    print("\n--- Merging LoRA weights for optimized inference ---")
    merged_backbone = backbone.merge_and_unload()
    print(f"Model type after merging: {type(merged_backbone)}")

    # You can now use `merged_backbone` as a regular Hugging Face model.
    # It has the LoRA adaptations baked into its weights.
    with torch.no_grad():
        merged_outputs = merged_backbone(**inputs)
        merged_cls_feature = merged_outputs.last_hidden_state[:, 0]
    
    print(f"Merged backbone CLS feature shape: {merged_cls_feature.shape}")
    # Verify that the output is (almost) identical
    assert torch.allclose(cls_feature, merged_cls_feature, atol=1e-5)
    print("✓ Output from PEFT model and merged model are identical.")