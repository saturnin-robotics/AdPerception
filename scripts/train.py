"""
train.py -- Transfer-learning training for CenterPoint detection head.

Freezes the Sonata-pretrained PTv3 encoder backbone and trains only:
  - backbone.proj   : linear 64→256 (adapts backbone features to detection)
  - det_head.neck   : SharedNeck 2× Conv2d-BN-ReLU (BEV feature enrichment)
  - det_head.heads  : DetectionHeads 6× 1×1 Conv2d (regression outputs)

Uses CenterPoint paper losses (focal + L1) with nuScenes mini GT annotations.

Usage:
    python train.py
    python train.py --epochs 30 --lr 5e-4
    python train.py --dataroot /path/to/nuscenes

Output:
    checkpoints/centerpoint.pth  (best epoch by training loss)
"""

import argparse
import os

import torch
import torch.optim as optim

from src.datasets.nuscenes_loader import NuScenesLiDARDataset
from src.models.pipeline import LiDARPerceptionPipeline
from src.models.heads.centerpoint_loss import centerpoint_loss

# -----defaults 
DATAROOT   = "/home/user/AdPerception/data/nuscenes"
SONATA_PTH = "checkpoints/sonata.pth"
CKPT_OUT   = "checkpoints/centerpoint.pth"

# BEV grid params -- must match pipeline voxel config
VOXEL_SIZE  = [0.2, 0.2, 0.4]
POINT_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
H = round((POINT_RANGE[4] - POINT_RANGE[1]) / VOXEL_SIZE[1])   # 512
W = round((POINT_RANGE[3] - POINT_RANGE[0]) / VOXEL_SIZE[0])   # 512


def build_pipeline(device: torch.device) -> LiDARPerceptionPipeline:
    pipeline = LiDARPerceptionPipeline(
        voxel_size     = VOXEL_SIZE,
        point_range    = POINT_RANGE,
        score_thresh   = 0.1,
        max_detections = 500,
        nms_kernel     = 7,
    ).to(device)

    if os.path.exists(SONATA_PTH):
        pipeline.load_sonata(SONATA_PTH)
    else:
        print(f"[warn] Sonata checkpoint not found: {SONATA_PTH}")

    return pipeline


def freeze_backbone(pipeline: LiDARPerceptionPipeline) -> None:
    """Freeze Sonata-pretrained PTv3 encoder weights."""
    for p in pipeline.backbone.backbone.parameters():
        p.requires_grad = False
    print("Backbone frozen (Sonata encoder weights preserved).")


def trainable_params(pipeline: LiDARPerceptionPipeline) -> list:
    """Return parameters to optimise (proj + det neck + det heads)."""
    return (
        list(pipeline.backbone.proj.parameters()) +
        list(pipeline.det_head.neck.parameters()) +
        list(pipeline.det_head.heads.parameters())
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CenterPoint head on nuScenes")
    parser.add_argument("--epochs",   type=int,   default=20,       help="training epochs")
    parser.add_argument("--lr",       type=float, default=1e-3,     help="initial learning rate")
    parser.add_argument("--dataroot", type=str,   default=DATAROOT, help="nuScenes root dir")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # -----dataset -----------------------------------------------------------
    dataset = NuScenesLiDARDataset(
        dataroot = args.dataroot,
        version  = "v1.0-mini",
        split    = "train",
        verbose  = False,
    )
    print(f"Dataset: {len(dataset)} samples (nuScenes mini)")

    # -----model --------------------------------------------------------------
    pipeline = build_pipeline(device)
    freeze_backbone(pipeline)

    params    = trainable_params(pipeline)
    n_params  = sum(p.numel() for p in params)
    print(f"Trainable params: {n_params:,}")

    # -----optimiser + scheduler ----------------------------------------------
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    scaler    = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # -----training loop ------------------------------------------------------
    best_loss = float("inf")

    header = (
        f"\n{'Epoch':>5} | {'Loss':>8} | {'Heatmap':>8} | "
        f"{'Offset':>8} | {'Dims':>8} | {'Rot':>8} | {'LR':>9}"
    )
    print(header)
    print("─" * len(header))

    for epoch in range(args.epochs):
        pipeline.train()
        # Keep frozen backbone in eval mode so its BatchNorm stats stay fixed
        pipeline.backbone.backbone.eval()

        epoch_loss  = 0.0
        epoch_parts: dict = {}
        n_samples   = 0

        for sample in dataset:
            pts       = sample["points"].to(device)
            gt_boxes  = sample["gt_boxes"].to(device)
            gt_labels = sample["gt_labels"].to(device)

            optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    raw = pipeline(pts, batch_size=1, decode=False)
                    loss, ld = centerpoint_loss(
                        raw["det_preds"], gt_boxes, gt_labels,
                        VOXEL_SIZE, POINT_RANGE, H, W,
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, max_norm=35.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                raw = pipeline(pts, batch_size=1, decode=False)
                loss, ld = centerpoint_loss(
                    raw["det_preds"], gt_boxes, gt_labels,
                    VOXEL_SIZE, POINT_RANGE, H, W,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=35.0)
                optimizer.step()

            epoch_loss += loss.item()
            for k, v in ld.items():
                epoch_parts[k] = epoch_parts.get(k, 0.0) + v
            n_samples += 1

        scheduler.step()

        n = max(n_samples, 1)
        avg      = epoch_loss / n
        hm_avg   = epoch_parts.get("heatmap",  0.0) / n
        off_avg  = epoch_parts.get("offset",   0.0) / n
        dims_avg = epoch_parts.get("dims",     0.0) / n
        rot_avg  = epoch_parts.get("rotation", 0.0) / n
        lr_now   = scheduler.get_last_lr()[0]

        print(
            f"{epoch+1:>5} | {avg:>8.4f} | {hm_avg:>8.4f} | "
            f"{off_avg:>8.4f} | {dims_avg:>8.4f} | {rot_avg:>8.4f} | {lr_now:>9.2e}"
        )

        if avg < best_loss:
            best_loss = avg
            torch.save({"state_dict": pipeline.state_dict()}, CKPT_OUT)
            print(f" saved {CKPT_OUT}  (loss={best_loss:.4f})")

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Checkpoint : {CKPT_OUT}")
    print(f"Inference  : python infer.py --predicted")


if __name__ == "__main__":
    main()
