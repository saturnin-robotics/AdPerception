"""
train.py -- Transfer-learning training for CenterPoint detection head.

Freezes the Sonata-pretrained PTv3 encoder (enc_stages + stem) and trains:
  - backbone.proj        : linear 64->256 (adapts backbone features)
  - backbone PTv3 decoder: produces BEV features tuned for detection
  - det_head.neck        : SharedNeck 2x Conv2d-BN-ReLU
  - det_head.heads       : DetectionHeads 6x 1x1 Conv2d

Features:
  - --no-amp  : disable mixed precision (for RTX Ada + spconv-cu120)
  - --wandb   : enable wandb experiment tracking
  - --resume  : resume from checkpoint after Jean Zay interruption
  - 3D augmentations: flip Y, rotation Z, scaling
  - Encoder-only freeze (PTv3 decoder trainable)
  - OneCycleLR stepping per batch
  - 85/15 train/val split with shuffle
  - Resume checkpoint saved after every epoch

Usage:
    python scripts/train.py --config local --no-amp          # RTX Ada local
    python scripts/train.py --config local --no-amp --wandb  # local + tracking
    python scripts/train.py --config jeanzay --wandb         # Jean Zay H100
    python scripts/train.py --config jeanzay --wandb \
        --resume checkpoints/resume.pth                      # Jean Zay resume

Output:
    checkpoints/centerpoint.pth  -- best checkpoint (val_loss)
    checkpoints/resume.pth       -- resume checkpoint (every epoch)
"""

import argparse
import os
import sys

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from src.datasets.nuscenes_loader import NuScenesLiDARDataset
from src.models.pipeline import LiDARPerceptionPipeline
from src.models.heads.centerpoint_loss import centerpoint_loss

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# =========================================================================
# Paths
# =========================================================================
SONATA_PTH  = "checkpoints/sonata.pth"
CKPT_OUT    = "checkpoints/centerpoint.pth"
CKPT_RESUME = "checkpoints/resume.pth"
DATAROOT    = "/home/user/AdPerception/data/nuscenes"

# =========================================================================
# Voxel configs
# =========================================================================
VOXEL_CONFIGS = {
    "local": {
        # RTX 1000 Ada (6 GB VRAM) -- BEV 1024x1024
        "voxel_size":           [0.1, 0.1, 0.2],
        "point_range":          [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        "max_points_per_voxel": 10,
        "max_voxels_train":     80_000,
        "max_voxels_test":      80_000,
    },
    "jeanzay": {
        # Official CenterPoint nuScenes config -- BEV 1440x1440
        # Reference: OpenPCDet cbgs_voxel0075_res3d_centerpoint.yaml
        "voxel_size":           [0.075, 0.075, 0.2],
        "point_range":          [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
        "max_points_per_voxel": 10,
        "max_voxels_train":     120_000,
        "max_voxels_test":      160_000,
    },
}


def get_bev_dims(cfg: dict) -> tuple[int, int]:
    """Compute BEV grid height and width from voxel config."""
    pr = cfg["point_range"]
    vs = cfg["voxel_size"]
    H  = round((pr[4] - pr[1]) / vs[1])
    W  = round((pr[3] - pr[0]) / vs[0])
    return H, W


# =========================================================================
# 3D augmentation
# =========================================================================

def augment_points_and_boxes(
    pts:      torch.Tensor,
    gt_boxes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Standard CenterPoint 3D augmentations applied per sample.

    pts      : (N, 4) -- x, y, z, intensity
    gt_boxes : (M, 9) -- x, y, z, l, w, h, yaw, vx, vy
               (M, 7) also accepted -- no velocity columns

    Returns cloned augmented tensors.
    """
    pts      = pts.clone()
    gt_boxes = gt_boxes.clone()
    has_vel  = gt_boxes.shape[1] > 7

    # 1. Random flip along Y axis (50% probability)
    if torch.rand(1).item() > 0.5:
        pts[:, 1]      = -pts[:, 1]
        gt_boxes[:, 1] = -gt_boxes[:, 1]
        gt_boxes[:, 6] = -gt_boxes[:, 6]
        if has_vel:
            gt_boxes[:, 8] = -gt_boxes[:, 8]

    # 2. Random rotation around Z axis in [-pi/4, +pi/4]
    angle = (torch.rand(1).item() - 0.5) * (3.14159265 / 2.0)
    cos_a = torch.tensor(angle).cos()
    sin_a = torch.tensor(angle).sin()

    x_pts = cos_a * pts[:, 0] - sin_a * pts[:, 1]
    y_pts = sin_a * pts[:, 0] + cos_a * pts[:, 1]
    pts[:, 0], pts[:, 1] = x_pts, y_pts

    x_box = cos_a * gt_boxes[:, 0] - sin_a * gt_boxes[:, 1]
    y_box = sin_a * gt_boxes[:, 0] + cos_a * gt_boxes[:, 1]
    gt_boxes[:, 0], gt_boxes[:, 1] = x_box, y_box
    gt_boxes[:, 6] = gt_boxes[:, 6] + angle

    if has_vel:
        vx_new = cos_a * gt_boxes[:, 7] - sin_a * gt_boxes[:, 8]
        vy_new = sin_a * gt_boxes[:, 7] + cos_a * gt_boxes[:, 8]
        gt_boxes[:, 7], gt_boxes[:, 8] = vx_new, vy_new

    # 3. Random scaling in [0.95, 1.05]
    scale            = 0.95 + torch.rand(1).item() * 0.10
    pts[:, :3]       = pts[:, :3] * scale
    gt_boxes[:, :3]  = gt_boxes[:, :3] * scale
    gt_boxes[:, 3:6] = gt_boxes[:, 3:6] * scale

    return pts, gt_boxes


# =========================================================================
# Model helpers
# =========================================================================

def build_pipeline(device: torch.device, cfg: dict) -> LiDARPerceptionPipeline:
    """Build pipeline with voxel params from selected config."""
    pipeline = LiDARPerceptionPipeline(
        voxel_size           = cfg["voxel_size"],
        point_range          = cfg["point_range"],
        max_points_per_voxel = cfg["max_points_per_voxel"],
        max_voxels           = cfg["max_voxels_train"],
        score_thresh         = 0.1,
        max_detections       = 500,
        nms_kernel           = 7,
    ).to(device)

    if os.path.exists(SONATA_PTH):
        pipeline.load_sonata(SONATA_PTH)
        print(f"Sonata weights loaded from {SONATA_PTH}")
    else:
        print(f"[warn] Sonata checkpoint not found at {SONATA_PTH}")
        print("       Training from random weights.")

    return pipeline


def freeze_encoder(pipeline: LiDARPerceptionPipeline) -> None:
    """
    Freeze only the Sonata encoder (enc_stages + stem).
    PTv3 decoder remains trainable to adapt BEV features for detection.
    """
    n_frozen = 0
    n_total  = 0
    for name, param in pipeline.backbone.backbone.named_parameters():
        n_total += 1
        if "enc_stages" in name or "stem" in name:
            param.requires_grad = False
            n_frozen += 1
    print(f"Encoder frozen: {n_frozen}/{n_total} backbone params.")


def trainable_params(pipeline: LiDARPerceptionPipeline) -> list:
    """
    Collect parameters that should receive gradients:
      - backbone.proj     (feature dimension adapter)
      - PTv3 decoder      (BEV feature generation)
      - det_head.neck     (BEV enrichment)
      - det_head.heads    (regression outputs)
    """
    decoder_params = [
        p for p in pipeline.backbone.backbone.parameters()
        if p.requires_grad
    ]
    return (
        list(pipeline.backbone.proj.parameters())
        + decoder_params
        + list(pipeline.det_head.neck.parameters())
        + list(pipeline.det_head.heads.parameters())
    )


# =========================================================================
# Validation
# =========================================================================

def run_validation(
    pipeline:  LiDARPerceptionPipeline,
    val_dataset,
    device:    torch.device,
    voxel_cfg: dict,
    H:         int,
    W:         int,
) -> float:
    """Single pass over the validation set. Returns average val_loss."""
    pipeline.eval()
    val_loss = 0.0

    with torch.no_grad():
        for sample in val_dataset:
            pts       = sample["points"].to(device)
            gt_boxes  = sample["gt_boxes"].to(device)
            gt_labels = sample["gt_labels"].to(device)

            raw = pipeline(pts, batch_size=1, decode=False)
            loss, _ = centerpoint_loss(
                raw["det_preds"], gt_boxes, gt_labels,
                voxel_cfg["voxel_size"],
                voxel_cfg["point_range"],
                H, W,
            )
            val_loss += loss.item()

    return val_loss / max(len(val_dataset), 1)


# =========================================================================
# Checkpoint utilities
# =========================================================================

def save_resume_checkpoint(
    path:          str,
    epoch:         int,
    pipeline:      LiDARPerceptionPipeline,
    optimizer:     torch.optim.Optimizer,
    scheduler,
    scaler,
    best_val_loss: float,
    voxel_cfg:     dict,
    args:          argparse.Namespace,
) -> None:
    """
    Save full training state after each epoch.
    Enables resuming after Jean Zay maintenance interruptions.
    Overwrites previous resume checkpoint -- only latest epoch kept.
    """
    torch.save({
        "epoch":         epoch,
        "state_dict":    pipeline.state_dict(),
        "optimizer":     optimizer.state_dict(),
        "scheduler":     scheduler.state_dict(),
        "scaler":        scaler.state_dict() if scaler is not None else None,
        "best_val_loss": best_val_loss,
        "voxel_cfg":     voxel_cfg,
        "args":          vars(args),
    }, path)


def load_resume_checkpoint(
    path:      str,
    pipeline:  LiDARPerceptionPipeline,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    device:    torch.device,
) -> tuple[int, float]:
    """
    Load full training state from a resume checkpoint.
    Returns (start_epoch, best_val_loss).
    """
    ckpt = torch.load(path, map_location=device)
    pipeline.load_state_dict(ckpt["state_dict"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    start_epoch   = ckpt["epoch"]
    best_val_loss = ckpt["best_val_loss"]
    print(f"Resumed from epoch {start_epoch} | best_val_loss={best_val_loss:.4f}")
    return start_epoch, best_val_loss


# =========================================================================
# Main
# =========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train CenterPoint head on nuScenes"
    )
    parser.add_argument("--epochs",   type=int,   default=20)
    parser.add_argument("--lr",       type=float, default=1e-2)
    parser.add_argument("--dataroot", type=str,   default=DATAROOT)
    parser.add_argument(
        "--version", type=str, default="v1.0-mini",
        choices=["v1.0-mini", "v1.0-trainval"],
    )
    parser.add_argument(
        "--config", type=str, default="local",
        choices=["local", "jeanzay"],
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--no-amp",
        action  = "store_true",
        default = False,
        help    = "disable AMP mixed precision (required for RTX Ada + spconv-cu120)",
    )
    parser.add_argument(
        "--wandb",
        action  = "store_true",
        default = False,
        help    = "enable wandb experiment tracking",
    )
    parser.add_argument(
        "--resume",
        type    = str,
        default = None,
        help    = "path to resume checkpoint (e.g. checkpoints/resume.pth)",
    )
    
    args = parser.parse_args()

    # Validate wandb
    use_wandb = args.wandb
    if use_wandb and not WANDB_AVAILABLE:
        print("[warn] wandb not installed -- run: pip install wandb")
        use_wandb = False

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    voxel_cfg = VOXEL_CONFIGS[args.config]
    H, W      = get_bev_dims(voxel_cfg)
    use_amp   = (device.type == "cuda") and (not args.no_amp)

    print(f"Device     : {device}")
    print(f"Config     : {args.config}")
    print(f"Voxel size : {voxel_cfg['voxel_size']} m")
    print(f"BEV grid   : {H} x {W}")
    print(f"Max voxels : {voxel_cfg['max_voxels_train']} (train)")
    print(f"AMP        : {'enabled' if use_amp else 'disabled (--no-amp)'}")
    print(f"Wandb      : {'enabled' if use_wandb else 'disabled'}")

    # =========================================================================
    # Wandb init
    # =========================================================================
    if use_wandb:
        run_id = f"adperc_{args.config}_{args.version}_lr{args.lr}"
        wandb.init(
            project = "adperception",
            name    = f"{args.config}_lr{args.lr}_ep{args.epochs}_{args.version}",
            id      = run_id,
            resume  = "allow",
            config  = {
                "lr":                   args.lr,
                "epochs":               args.epochs,
                "version":              args.version,
                "config":               args.config,
                "workers":              args.workers,
                "amp":                  use_amp,
                "voxel_size":           voxel_cfg["voxel_size"],
                "point_range":          voxel_cfg["point_range"],
                "max_points_per_voxel": voxel_cfg["max_points_per_voxel"],
                "max_voxels_train":     voxel_cfg["max_voxels_train"],
                "bev_H":                H,
                "bev_W":                W,
            },
            dir = os.environ.get("WANDB_DIR", "wandb"),
        )
        os.makedirs(os.environ.get("WANDB_DIR", "wandb"), exist_ok=True)

    # =========================================================================
    # Dataset
    # =========================================================================
    full_dataset = NuScenesLiDARDataset(
        dataroot = args.dataroot,
        version  = args.version,
        split    = "train",
        verbose  = False,
    )

    n_val   = max(1, int(0.15 * len(full_dataset)))
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size  = 1,
        shuffle     = True,
        num_workers = args.workers,
        pin_memory  = device.type == "cuda",
        collate_fn  = lambda x: x[0],
    )

    print(f"Train      : {n_train} samples")
    print(f"Val        : {n_val} samples")

    # =========================================================================
    # Model + optimizer + scheduler
    # =========================================================================
    pipeline = build_pipeline(device, voxel_cfg)
    freeze_encoder(pipeline)

    params   = trainable_params(pipeline)
    n_params = sum(p.numel() for p in params)
    print(f"Trainable  : {n_params:,} params")

    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr           = args.lr,
        epochs           = args.epochs,
        steps_per_epoch  = len(train_loader),
        pct_start        = 0.3,
        div_factor       = 25.0,
        final_div_factor = 1e4,
    )
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # =========================================================================
    # Resume
    # =========================================================================
    start_epoch   = 0
    best_val_loss = float("inf")

    if args.resume:
        if os.path.exists(args.resume):
            start_epoch, best_val_loss = load_resume_checkpoint(
                args.resume, pipeline, optimizer, scheduler, scaler, device
            )
        else:
            print(f"[warn] Resume checkpoint not found at {args.resume} -- starting fresh.")

    # =========================================================================
    # Training loop
    # =========================================================================
    os.makedirs(os.path.dirname(CKPT_OUT),    exist_ok=True)
    os.makedirs(os.path.dirname(CKPT_RESUME), exist_ok=True)

    header = (
        f"\n{'Epoch':>5} | {'TrainLoss':>9} | {'ValLoss':>9} | "
        f"{'Heatmap':>8} | {'Offset':>8} | {'Dims':>8} | "
        f"{'Rot':>8} | {'LR':>9}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    for epoch in range(start_epoch, args.epochs):

        # =====================================================================
        # Train
        # =====================================================================
        pipeline.train()
        # Keep frozen encoder BatchNorm stats fixed
        pipeline.backbone.backbone.eval()
        # Re-enable train mode for decoder submodules only
        for name, module in pipeline.backbone.backbone.named_modules():
            if "dec_stages" in name or "dec" in name:
                module.train()

        epoch_loss              = 0.0
        epoch_parts: dict[str, float] = {}
        n_batches               = 0
        global_step             = epoch * len(train_loader)

        for sample in train_loader:
            pts       = sample["points"].to(device)
            gt_boxes  = sample["gt_boxes"].to(device)
            gt_labels = sample["gt_labels"].to(device)

            pts, gt_boxes = augment_points_and_boxes(pts, gt_boxes)
            optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                # AMP path -- H100 Jean Zay
                with torch.cuda.amp.autocast():
                    raw = pipeline(pts, batch_size=1, decode=False)
                    loss, ld = centerpoint_loss(
                        raw["det_preds"], gt_boxes, gt_labels,
                        voxel_cfg["voxel_size"],
                        voxel_cfg["point_range"],
                        H, W,
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, max_norm=35.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # fp32 path -- RTX Ada local or --no-amp
                raw = pipeline(pts, batch_size=1, decode=False)
                loss, ld = centerpoint_loss(
                    raw["det_preds"], gt_boxes, gt_labels,
                    voxel_cfg["voxel_size"],
                    voxel_cfg["point_range"],
                    H, W,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=35.0)
                optimizer.step()

            scheduler.step()

            epoch_loss += loss.item()
            for k, v in ld.items():
                epoch_parts[k] = epoch_parts.get(k, 0.0) + v
            n_batches   += 1
            global_step += 1

            # Wandb batch log every 50 batches
            if use_wandb and n_batches % 50 == 0:
                wandb.log({
                    "batch/loss":    loss.item(),
                    "batch/lr":      scheduler.get_last_lr()[0],
                    "batch/heatmap": ld.get("heatmap",  0.0),
                    "batch/offset":  ld.get("offset",   0.0),
                    "batch/dims":    ld.get("dims",     0.0),
                    "batch/rot":     ld.get("rotation", 0.0),
                }, step=global_step)

        # =====================================================================
        # Validation
        # =====================================================================
        val_loss = run_validation(
            pipeline, val_dataset, device, voxel_cfg, H, W
        )

        n        = max(n_batches, 1)
        avg      = epoch_loss / n
        hm_avg   = epoch_parts.get("heatmap",  0.0) / n
        off_avg  = epoch_parts.get("offset",   0.0) / n
        dims_avg = epoch_parts.get("dims",     0.0) / n
        rot_avg  = epoch_parts.get("rotation", 0.0) / n
        lr_now   = scheduler.get_last_lr()[0]

        # Wandb epoch log
        if use_wandb:
            wandb.log({
                "epoch/train_loss": avg,
                "epoch/val_loss":   val_loss,
                "epoch/heatmap":    hm_avg,
                "epoch/offset":     off_avg,
                "epoch/dims":       dims_avg,
                "epoch/rot":        rot_avg,
                "epoch/lr":         lr_now,
                "epoch":            epoch + 1,
            }, step=global_step)

        # Save best checkpoint
        saved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch":      epoch + 1,
                "state_dict": pipeline.state_dict(),
                "val_loss":   best_val_loss,
                "voxel_cfg":  voxel_cfg,
            }, CKPT_OUT)
            saved = f"  --> saved (val={best_val_loss:.4f})"

            if use_wandb:
                artifact = wandb.Artifact(
                    name = f"centerpoint-best-{args.config}",
                    type = "model",
                )
                artifact.add_file(CKPT_OUT)
                wandb.log_artifact(artifact)

        # Save resume checkpoint (overwrites every epoch)
        save_resume_checkpoint(
            path          = CKPT_RESUME,
            epoch         = epoch + 1,
            pipeline      = pipeline,
            optimizer     = optimizer,
            scheduler     = scheduler,
            scaler        = scaler,
            best_val_loss = best_val_loss,
            voxel_cfg     = voxel_cfg,
            args          = args,
        )

        print(
            f"{epoch+1:>5} | {avg:>9.4f} | {val_loss:>9.4f} | "
            f"{hm_avg:>8.4f} | {off_avg:>8.4f} | {dims_avg:>8.4f} | "
            f"{rot_avg:>8.4f} | {lr_now:>9.2e}"
            + saved
        )

        pipeline.train()

    # =========================================================================
    # End
    # =========================================================================
    print(sep)
    print(f"Training complete.")
    print(f"Best val_loss : {best_val_loss:.4f}")
    print(f"Checkpoint    : {CKPT_OUT}")
    print(f"Inference     : python scripts/infer.py --checkpoint {CKPT_OUT}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
