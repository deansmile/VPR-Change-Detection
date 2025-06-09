# ================================================================
#  ExPLoRA-style self-supervised adaptation of DINOv2 with LoRA
#  https://paperswithcode.com/paper/explora-parameter-efficient-extended-pre
#  Official code not available yet but this is our attempt
#  Requirements: pip install torch torchvision timm transformers peft
# ================================================================


## cd VPR-Change-Detection
## python -m src.dinov2_lora

import math, random, copy, collections, os
from typing import List

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T
from transformers import Dinov2Model, Dinov2Config
from peft import LoraConfig, get_peft_model

#dataset related
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor



# sys.path.append("../../../")


dataset_path = "../Datasets/VL-CMU-CD"



class CMUChangeDataset(Dataset):
    """
    root/train/{t0,t1,mask}            ← split = "train" or "test"
    ├── 00005_1_00_0.png               ← t0 image
    ├── …
    """
    def __init__(self,
                 root: str,
                 split: str = "train",
                 img_size: int = 224,  # dino features work best on 224x224
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


class DinoCrops:
    """Return list of PIL images: 2 global (224) + 8 local (98)."""
    def __init__(self):
        normalize = T.Normalize([0.485,0.456,0.406],[0.228,0.224,0.225])

        # global 224 crops (50-100 % of image)
        self.global_t = T.Compose([
            T.RandomResizedCrop(224, scale=(0.5,1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.4,0.4,0.4,0.1),
            T.RandomGrayscale(0.2),
            T.GaussianBlur(23, sigma=(0.1,2.0)),
            T.ToTensor(), normalize])
        # local 98 crops (5-32 %)
        self.local_t  = T.Compose([
            T.RandomResizedCrop(98,  scale=(0.05,0.32)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.4,0.4,0.4,0.1),
            T.RandomGrayscale(0.2),
            T.GaussianBlur(23, sigma=(0.1,2.0)),
            T.ToTensor(), normalize])

    def __call__(self, img):
        crops = [self.global_t(img) for _ in range(2)]
        crops += [self.local_t(img)  for _ in range(8)]
        return crops                         # list[Tensor]

# -----------------------------------------------------------------
# 3. DINO/iBOT head + prototypes
# -----------------------------------------------------------------
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
        self.apply(self._init_weights)
        self.prototypes = nn.Linear(bottleneck_dim, n_prototypes, bias=False)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)

    def forward(self, x):                   # (B,* ,D)
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        return self.prototypes(x)           # logits

# -----------------------------------------------------------------
# 4. Student + teacher backbones with LoRA
# -----------------------------------------------------------------
def build_backbones():
    # Using dinov2-base, which has 12 layers (0-11)
    student = Dinov2Model.from_pretrained("facebook/dinov2-base")
    teacher = copy.deepcopy(student).eval()
    for p in teacher.parameters(): # freeze teacher
        p.requires_grad_(False)

    # (a) unfreeze last encoder block + all LayerNorms
    for n,p in student.named_parameters():
        p.requires_grad = n.startswith("encoder.layer.11")
    for m in student.modules():
        if isinstance(m, nn.LayerNorm):
            m.weight.requires_grad_(True)
            m.bias.requires_grad_(True)

    # (b) LoRA only on query & value of *frozen* blocks
    peft_cfg = LoraConfig(
        r=64, lora_alpha=64, lora_dropout=0.05,
        task_type="FEATURE_EXTRACTION",
        target_modules=["query", "value"])
    student = get_peft_model(student, peft_cfg)
    return student, teacher

# -----------------------------------------------------------------
# 5. Loss utilities
# -----------------------------------------------------------------
@torch.no_grad()
def update_center(center, teacher_logits, momentum=0.9):
    """Update the running center of teacher outputs."""
    batch_center = teacher_logits.mean(dim=0, keepdim=True)
    return center * momentum + batch_center * (1 - momentum)

# -----------------------------------------------------------------
# 6. Training
# -----------------------------------------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 32
    epochs = 100           # ≈200 k iters on ImageNet-1M;
    lr_base  = 2e-3

    # -------- data --------
    cropper = DinoCrops()

    def collate(batch):
        # batch = list[ dict(t0,t1,mask,filename) ]
        images = [b["t0"] for b in batch]             # List of PIL Images
        # Apply multi-crop augmentation to each image
        multi_crops_list = [cropper(img) for img in images]
        # Flatten the list of lists into a single list of tensors
        crops = [crop for sublist in multi_crops_list for crop in sublist]
        return torch.stack(crops), len(images) # (B*10, 3, H, W), B

    train_ds       = CMUChangeDataset(dataset_path, split="train")
    train_loader   = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)


    # -------- models -------------------------
    student, teacher = build_backbones()
    head_s = DinoHead().to(device)
    head_t = copy.deepcopy(head_s).eval().to(device)

    student, teacher = student.to(device), teacher.to(device)
    student.print_trainable_parameters()

    # -------- optimiser & schedulers --------
    iters_per_epoch = len(train_loader)
    total_iters = epochs * iters_per_epoch
    params_to_train = list(filter(lambda p: p.requires_grad, student.parameters())) + list(head_s.parameters())
    opt = torch.optim.AdamW(params_to_train, lr=lr_base, betas=(0.9,0.999), weight_decay=1e-4)

    def cosine_sched(base, final, it, tot):
        cos = 0.5 * (1 + math.cos(math.pi * it / tot))
        return final + (base - final) * cos
    
    # -------- training loop --------
    center = torch.zeros(1, 65_536, device=device)
    for epoch in range(epochs):
        for i, (crops, b_actual) in enumerate(train_loader):
            it = epoch * iters_per_epoch + i
            if it >= total_iters: break

            # scheduler steps
            temp_s = cosine_sched(0.1, 0.07, it, total_iters)
            temp_t = cosine_sched(0.04, 0.07, it, total_iters)
            ema_m  = cosine_sched(0.994, 1.0, it, total_iters)
            lr     = cosine_sched(lr_base, 1e-5, it, total_iters)
            for pg in opt.param_groups: pg["lr"] = lr

            crops = crops.to(device, non_blocking=True)
            # split views: first 2*B global for teacher, all 10*B for student
            n_global = 2 * b_actual
            imgs_t   = crops[:n_global]
            imgs_s   = crops

            # === FORWARD PASSES (with teacher on old weights) ===
            feat_s = student(pixel_values=imgs_s).last_hidden_state     # (10B, N, D)
            st_cls = head_s(feat_s[:,0])                                # (10B, P) -> Student CLS logits

            with torch.no_grad():
                feat_t = teacher(pixel_values=imgs_t).last_hidden_state # (2B, N, D)
                te_cls = head_t(feat_t[:,0])                            # (2B, P) -> Teacher CLS logits

            # 1. Prepare teacher targets: apply centering and sharpening (softmax with low temp)
            with torch.no_grad():
                te_targets = F.softmax((te_cls - center) / temp_t, dim=-1) # (2B, P)

            # 2. Prepare student predictions: apply sharpening (log_softmax with higher temp)
            st_log_preds = F.log_softmax(st_cls / temp_s, dim=-1)      # (10B, P)

            # 3. Compute cross-entropy loss for all student vs. teacher view pairs
            # Reshape for broadcasting:
            st_views = st_log_preds.view(b_actual, 10, -1) # (B, 10_student_views, D_proj)
            te_views = te_targets.view(b_actual, 2, -1)   # (B, 2_teacher_views, D_proj)

            # Each of the 10 student views is compared to each of the 2 teacher views
            # (B, 10, 1, D) * (B, 1, 2, D) -> sum over D -> (B, 10, 2)
            loss_matrix = (st_views.unsqueeze(2) * te_views.unsqueeze(1)).sum(dim=-1)
            loss = -loss_matrix.mean()

            # === BACKWARD PASS & OPTIMIZER STEP ===
            opt.zero_grad(set_to_none=True)
            loss.backward()
            # Optional: gradient clipping can help stability
            torch.nn.utils.clip_grad_norm_(params_to_train, 5.0)
            opt.step()

            # === EMA & CENTER UPDATES ===
            with torch.no_grad():
                # Update teacher backbone
                for ps, pt in zip(student.parameters(), teacher.parameters()):
                    if ps.requires_grad:
                        pt.data.mul_(ema_m).add_(ps.data, alpha=1 - ema_m)
                # Update teacher head
                for hs, ht in zip(head_s.parameters(), head_t.parameters()):
                    ht.data.mul_(ema_m).add_(hs.data, alpha=1 - ema_m)
                # Update running center
                center = update_center(center, te_cls.detach(), momentum=0.9)

            if i % 20 == 0:
                print(f"E{epoch:03d} it {it:06d}/{total_iters}  "
                    f"loss={loss.item():.4f}  T_s={temp_s:.3f}  lr={lr:.1e}")

    # -------- save LoRA adapters + head --------
    save_dir = "explora_lora_corrected"
    os.makedirs(save_dir, exist_ok=True)
    student.save_pretrained(f"{save_dir}/student_lora")
    torch.save(head_s.state_dict(), f"{save_dir}/dino_head.pt")
    print(f"✓ Saved PEFT adapters and projection head to {save_dir}/")



if __name__ == '__main__':
    main()

# ---------8. Load the model ------------------------------------

# from peft import PeftModel

# backbone = get_peft_model(Dinov2Model.from_pretrained("facebook/dinov2-base"),
#                           LoraConfig(...)).from_pretrained("explora_lora/student_lora")