# ================================================================
#  ExPLoRA-style self-supervised adaptation of DINOv2 with LoRA
#  https://paperswithcode.com/paper/explora-parameter-efficient-extended-pre
#  Official code not available yet but this is our attempt
#  Requirements: pip install torch torchvision timm transformers peft
# ================================================================


## cd X:\03_code\vpr\VPR-Change-Detection
## python -m src.models.dinov2_lora

import math, torch, copy, timm, collections
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from transformers import AutoImageProcessor, Dinov2Model
# from transformers import DinoVisionModel, DinoImageProcessor
from peft import LoraConfig, get_peft_model

from pathlib import Path
from PIL import Image




# sys.path.append("../../../")


dataset_path = "../../Datasets/VL-CMU-CD"

# -------- 1. Data ------------------------------------------------

# aug = T.Compose([
#     T.RandomResizedCrop(224, scale=(0.2, 1.0)),
#     T.RandomHorizontalFlip(),
#     T.ColorJitter(0.4, 0.4, 0.4, 0.1),
#     T.RandomGrayscale(p=0.2),
#     T.ToTensor(),
#     T.Normalize(mean=[0.485, 0.456, 0.406],
#                 std=[0.228, 0.224, 0.225]),
# ])

class CMUChangeDataset(Dataset):
    """
    root/train/{t0,t1,mask}            ← split = "train" or "test"
    ├── 00005_1_00_0.png               ← t0 image
    ├── …
    """
    def __init__(self,
                 root: str,
                 split: str = "train",
                 img_size: int = 224,
                 processor_name: str = "facebook/dinov2-base"):
        super().__init__()

        self.split   = split.lower()
        self.root    = Path(root) / self.split          # e.g. /data/vlcmucd/train
        self.t0_dir  = self.root / "t0"
        self.t1_dir  = self.root / "t1"
        self.mask_dir = self.root / "mask"

        # ---- collect & sort filenames so they line up by index
        self.fnames = sorted(f.name for f in self.t0_dir.glob("*.png"))

        # ---- lightweight image pre-processing (same as DINOv2)
        self.processor = AutoImageProcessor.from_pretrained(processor_name, use_fast=True)
        self.to_tensor = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),                       # -> [0,1]
            # binary mask: leave as float {0.,1.}
        ])

        # resize / crop for t0 & t1; use HF processor so
        # it automatically normalises to DINOv2 mean / std
        self.image_size = img_size

    def __len__(self) -> int:
        return len(self.fnames)

    def _load_rgb(self, path: Path) -> torch.Tensor:
        """Load & preprocess one RGB frame."""
        img = Image.open(path).convert("RGB")
        # AutoImageProcessor expects list/np/PIL, returns dict
        return self.processor(
            images=img,
            do_resize=True,
            size={"height": self.image_size, "width": self.image_size},  # Explicitly set both dimensions
            return_tensors="pt"
        )["pixel_values"][0]          # (3, H, W) float32

    def _load_mask(self, path: Path) -> torch.Tensor:
        """Load binary mask -> float tensor 0/1, same spatial size."""
        m = Image.open(path).convert("L")
        m = self.to_tensor(m)                         # (1,H,W)
        m = (m > 0.5).float()                         # binarise
        return m[0]                                   # (H,W)

    def __getitem__(self, idx: int):
        fname = self.fnames[idx]
        t0 = self._load_rgb(self.t0_dir / fname)
        t1 = self._load_rgb(self.t1_dir / fname)
        mask = self._load_mask(self.mask_dir / fname)

        return {
            "t0": t0,          # (3,H,W) torch.float32
            "t1": t1,          # (3,H,W)
            "mask": mask,      # (H,W)
            "filename": fname
        }


# # -----------------------------------------------------------------
# # quick smoke test
# # -----------------------------------------------------------------
# if __name__ == "__main__":
#     from torch.utils.data import DataLoader

#     ds = CMUChangeDataset(dataset_path, split="train")
#     print("Dataset length:", len(ds))

#     dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=4)

#     batch = next(iter(dl))
#     for k, v in batch.items():
#         print(k, type(v), v.shape)

if __name__ == '__main__':  
    train_dataset  = CMUChangeDataset(dataset_path, split="train")
    train_loader   = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)


    # -------- 2. Student & teacher backbone -------------------------
    student = Dinov2Model.from_pretrained("facebook/dinov2-base")
    teacher = copy.deepcopy(student).eval()
    for p in teacher.parameters(): p.requires_grad_(False)

    # 2a. Unfreeze *only* last encoder block (U = {L})
    for n, p in student.named_parameters():
        p.requires_grad = n.startswith("encoder.layers.11")

    # 2b. Always unfreeze LayerNorm affine params
    for m in student.modules():
        if isinstance(m, nn.LayerNorm):  # tiny
            m.weight.requires_grad_(True)
            m.bias.requires_grad_(True)

    # -------- 3. Inject LoRA into frozen QKV linears ---------------
    peft_cfg = LoraConfig(
        task_type="FEATURE_EXTRACTION",      # doesn't matter—just disables text-specific rules
        r=64, lora_alpha=64, lora_dropout=0.05,
        target_modules = ["query", "key", "value"],
        fan_in_fan_out=False
    )
    student = get_peft_model(student, peft_cfg)
    # print(student.print_trainable_parameters())


    # -------- 4. Dino-style SSL loss --------------------------------
    # Small helper to extract patch + CLS embeddings
    def projector(z):                   # 1024-dim to 65536 proto like DINOv2 (simplified)
        return nn.functional.normalize(z, dim=-1)

    temperature_student = 0.1
    temperature_teacher = 0.04
    center = torch.zeros(1, 65536, device="cuda")

    def dino_loss(x_student, x_teacher):
        global center
        # x: (B, P+1, D)
        q_s = projector(x_student) / temperature_student
        k_t = projector(x_teacher.detach() - center) / temperature_teacher
        student_logits = q_s.mean(1)      # global
        teacher_logits = k_t.mean(1)
        loss = torch.sum(
            -nn.functional.log_softmax(student_logits, dim=-1)
            * nn.functional.softmax(teacher_logits, dim=-1), dim=-1
        ).mean()
        # update running center
        center = center * 0.9 + k_t.mean(0, keepdim=True) * 0.1
        return loss

    # -------- 5. Optimizer & EMA ------------------------------------
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, student.parameters()),
                            lr=5e-4, weight_decay=1e-4)
    ema_momentum = 0.996

    # -------- 6. Training loop (manage 50 k images ≈ 10 epochs) -----
    device = "cuda"
    student, teacher = student.to(device), teacher.to(device)

    for epoch in range(10):
        for batch in train_loader:
            imgs = batch["t0"]
            imgs = imgs.to(device, non_blocking=True)

            # two views for student (weak+strong); teacher gets strong view only
            imgs_teacher = imgs
            imgs_student = imgs  # reuse; plug stronger aug pipeline if you like

            # forward
            feat_s = student(pixel_values=imgs_student, output_hidden_states=False).last_hidden_state
            with torch.no_grad():
                feat_t = teacher(pixel_values=imgs_teacher, output_hidden_states=False).last_hidden_state

            loss = dino_loss(feat_s, feat_t)

            opt.zero_grad()
            loss.backward()
            opt.step()

            # EMA update
            with torch.no_grad():
                ms, mt = student.parameters(), teacher.parameters()
                for ps, pt in zip(ms, mt):
                    pt.data.mul_(ema_momentum).add_(ps.data, alpha=1 - ema_momentum)

        print(f"Epoch {epoch:02d} – loss {loss.item():.4f}")

    # -------- 7. Save only ∆T adapters ------------------------------
    student.save_pretrained("dinov2_explora_lora")  # just 20-30 MB