"""
infer.py -- AdPerception inference + Rerun visualization.

Usage:
    python scripts/infer.py                                   # GT boxes + seg colors
    python scripts/infer.py --use-gt                          # GT boxes + height colors (no GPU)
    python scripts/infer.py --predicted --local               # trained head, local 6GB GPU
    python scripts/infer.py --predicted                       # trained head, H100 full res
    python scripts/infer.py --predicted --local --scene 2     # scene 2
    python scripts/infer.py --predicted --local --max-frames 50
    python scripts/infer.py --predicted --score-thresh 0.03   # lower threshold
"""

import os
import time
import argparse
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import torch
import numpy as np
from src.datasets.nuscenes_loader import NuScenesLiDARDataset
from src.models.pipeline import LiDARPerceptionPipeline
from src.tracking.simpletrack_wrapper import SimpleTrackWrapper
from src.visualization.rerun_viz import Visualizer

# All nuScenes detection classes (0-9) -- filtering done by score_thresh only
ALL_CLASSES = set(range(10))

# Ego vehicle footprint half-extents in meters (Renault Zoe 4.08m x 1.77m)
EGO_HALF_L = 2.04
EGO_HALF_W = 0.89

# Local voxel config for RTX 1000 Ada (6 GB VRAM) -- BEV 1024x1024
LOCAL_VOXEL_CFG = {
    "voxel_size":  [0.1, 0.1, 0.2],
    "point_range": [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
    "max_voxels":  80_000,
}


def _height_colors(pts_np: np.ndarray) -> np.ndarray:
    """Color LiDAR points by Z height: rainbow gradient from -2m (blue) to +4m (red)."""
    t = np.clip((pts_np[:, 2] + 2.0) / 6.0, 0.0, 1.0)
    stops = [0.0, 0.25, 0.5, 0.75, 1.0]
    r = np.interp(t, stops, [  0,   0,   0, 255, 255])
    g = np.interp(t, stops, [  0, 255, 255, 255,   0])
    b = np.interp(t, stops, [255, 255,   0,   0,   0])
    return np.stack([r, g, b], axis=1).astype(np.uint8)


def _gt_out(sample: dict, device: torch.device) -> dict:
    """
    Build pipeline-compatible output dict from nuScenes GT annotations.
    GT box format : (M, 7)  [cx, cy, cz, l, w, h, yaw]
    Output format : (M, 8)  [cx, cy, cz, l, w, h, sin(yaw), cos(yaw)]
    """
    gt   = sample["gt_boxes"]
    lbls = sample["gt_labels"]
    yaw  = gt[:, 6]
    boxes_8 = torch.cat(
        [gt[:, :6], torch.sin(yaw).unsqueeze(1), torch.cos(yaw).unsqueeze(1)],
        dim=1,
    )
    pt_colors = torch.from_numpy(_height_colors(sample["points"].numpy()))
    return {
        "boxes"        : boxes_8.to(device),
        "scores"       : torch.ones(len(gt), device=device),
        "labels"       : lbls.to(device),
        "velocity"     : torch.zeros(len(gt), 2, device=device),
        "point_colors" : pt_colors,
    }


def build_pipeline(
    args:   argparse.Namespace,
    device: torch.device,
) -> LiDARPerceptionPipeline:
    """
    Build pipeline with correct voxel config.

    Priority order:
      1. --local flag -> LOCAL_VOXEL_CFG (fits in 6GB VRAM)
      2. voxel_cfg saved in checkpoint (matches training exactly)
      3. LOCAL_VOXEL_CFG fallback if no checkpoint voxel_cfg
    """
    if args.local:
        voxel_size  = LOCAL_VOXEL_CFG["voxel_size"]
        point_range = LOCAL_VOXEL_CFG["point_range"]
        max_voxels  = LOCAL_VOXEL_CFG["max_voxels"]
        print("Voxel config: local (0.1m, BEV 1024x1024) -- fits 6GB VRAM")
    elif args.predicted and os.path.exists(args.checkpoint):
        ckpt      = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        cfg       = ckpt.get("voxel_cfg", None)
        if cfg is not None:
            voxel_size  = cfg["voxel_size"]
            point_range = cfg["point_range"]
            max_voxels  = cfg.get("max_voxels_test", 160_000)
            print(f"Voxel config from checkpoint: voxel_size={voxel_size}")
        else:
            voxel_size  = LOCAL_VOXEL_CFG["voxel_size"]
            point_range = LOCAL_VOXEL_CFG["point_range"]
            max_voxels  = LOCAL_VOXEL_CFG["max_voxels"]
            print("Voxel config: local fallback (no voxel_cfg in checkpoint)")
    else:
        voxel_size  = LOCAL_VOXEL_CFG["voxel_size"]
        point_range = LOCAL_VOXEL_CFG["point_range"]
        max_voxels  = LOCAL_VOXEL_CFG["max_voxels"]
        print("Voxel config: local default")

    pipeline = LiDARPerceptionPipeline(
        voxel_size           = voxel_size,
        point_range          = point_range,
        max_points_per_voxel = 10,
        max_voxels           = max_voxels,
        score_thresh         = args.score_thresh,
        max_detections       = 500,
        nms_kernel           = 7,
    ).to(device)

    pipeline.load_sonata("checkpoints/sonata.pth")

    if args.predicted:
        if os.path.exists(args.checkpoint):
            state    = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            pipeline.load_state_dict(state["state_dict"], strict=False)
            epoch    = state.get("epoch", "?")
            val_loss = state.get("val_loss", "?")
            print(f"Loaded checkpoint: {args.checkpoint} (epoch={epoch}, val_loss={val_loss:.4f})")
        else:
            print(f"[warn] Checkpoint not found at {args.checkpoint}")

    return pipeline


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = NuScenesLiDARDataset(
        dataroot = args.dataroot,
        version  = args.version,
        verbose  = False,
    )

    pipeline = None
    if not args.use_gt:
        pipeline = build_pipeline(args, device)
        pipeline.eval()
        print("Warming up CUDA kernels...")
        with torch.no_grad():
            _ = pipeline(dataset[0]["points"].to(device))
        print("Done.")

    # Read LiDAR sensor translation -- used to position ego box and suppress self-detections
    _s0    = dataset.nusc.get("sample", dataset.sample_tokens[0])
    _sd0   = dataset.nusc.get("sample_data", _s0["data"]["LIDAR_TOP"])
    _cal0  = dataset.nusc.get("calibrated_sensor", _sd0["calibrated_sensor_token"])
    ego_cx = float(_cal0["translation"][0])   # ~0.944 m forward of rear axle

    tracker = SimpleTrackWrapper(num_classes=10, max_trail_len=10)
    viz     = Visualizer(app_name="AdPerception", ego_cx=ego_cx)

    start_idx = args.scene * 10
    end_idx   = min(start_idx + args.max_frames, len(dataset)) if args.max_frames else len(dataset)

    if args.use_gt:
        mode = "GT boxes + height colors (no pipeline)"
    elif args.predicted:
        mode = f"Predicted boxes + seg colors (score_thresh={args.score_thresh}, {'local' if args.local else 'full-res'})"
    else:
        mode = "GT boxes + PTv3 Sonata seg colors"

    print(f"\nMode   : {mode}")
    print(f"Frames : {start_idx} -> {end_idx - 1}  ({end_idx - start_idx} frames)\n")
    print(f"{'Frame':>5} | {'Pts':>6} | {'Dets':>4} | {'Trk':>4} | {'Percep':>8} | {'Viz':>6}")
    print("-" * 52)

    prev_scene_token = None

    try:
        for i in range(start_idx, end_idx):
            sample      = dataset[i]
            scene_token = dataset.nusc.get("sample", sample["sample_token"])["scene_token"]

            # Reset tracker at scene boundary
            if scene_token != prev_scene_token:
                if prev_scene_token is not None:
                    print("--- new scene -- tracker reset ---")
                tracker.reset()
                prev_scene_token = scene_token

            t0 = time.perf_counter()

            if args.use_gt:
                out     = _gt_out(sample, device)
                pts_gpu = sample["points"]
            elif args.predicted:
                pts_gpu = sample["points"].to(device)
                with torch.no_grad():
                    out = pipeline(pts_gpu)
            else:
                pts_gpu = sample["points"].to(device)
                with torch.no_grad():
                    seg_out = pipeline(pts_gpu)
                gt_part = _gt_out(sample, device)
                out = {
                    "boxes"        : gt_part["boxes"],
                    "scores"       : gt_part["scores"],
                    "labels"       : gt_part["labels"],
                    "velocity"     : gt_part["velocity"],
                    "point_colors" : seg_out["point_colors"],
                    "point_labels" : seg_out.get("point_labels"),
                }

            t1 = time.perf_counter()

            # Keep all classes -- score_thresh handles filtering
            labels_np = out["labels"].cpu().numpy()
            dyn_mask  = np.isin(labels_np, list(ALL_CLASSES))

            boxes_dyn    = out["boxes"][dyn_mask]
            scores_dyn   = out["scores"][dyn_mask]
            labels_dyn   = out["labels"][dyn_mask]
            velocity_dyn = out["velocity"][dyn_mask]

            # Suppress detections whose center falls inside the ego vehicle footprint
            boxes_xy   = boxes_dyn.cpu()
            not_in_ego = ~(
                ((boxes_xy[:, 0] - ego_cx).abs() < EGO_HALF_L) &
                (boxes_xy[:, 1].abs()             < EGO_HALF_W)
            )
            boxes_dyn    = boxes_dyn[not_in_ego]
            scores_dyn   = scores_dyn[not_in_ego]
            labels_dyn   = labels_dyn[not_in_ego]
            velocity_dyn = velocity_dyn[not_in_ego]

            tracks = tracker.update(
                boxes    = boxes_dyn,
                scores   = scores_dyn,
                labels   = labels_dyn,
                velocity = velocity_dyn,
            )

            t2 = time.perf_counter()

            # Show GT overlay in --predicted mode for visual comparison
            show_gt = args.predicted
            viz.update(
                pts_gpu, out, tracks=tracks, frame_idx=i,
                gt_boxes  = sample["gt_boxes"]  if show_gt else None,
                gt_labels = sample["gt_labels"] if show_gt else None,
            )

            t3 = time.perf_counter()

            print(
                f"{i:>5} | {sample['points'].shape[0]:>6} | "
                f"{int(not_in_ego.sum()):>4} | {len(tracks):>4} | "
                f"{1000*(t1-t0):>6.0f}ms | {1000*(t3-t2):>4.0f}ms"
            )

    except KeyboardInterrupt:
        print("\nStopped -- Rerun viewer stays open for review")
    finally:
        viz.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AdPerception inference + Rerun viz")
    parser.add_argument("--use-gt",      action="store_true",
                        help="GT boxes + height colors (no GPU)")
    parser.add_argument("--predicted",   action="store_true",
                        help="run trained CenterPoint head")
    parser.add_argument("--local",       action="store_true", default=False,
                        help="force local voxel config (0.1m BEV 1024x1024) for 6GB GPU")
    parser.add_argument("--scene",       type=int,   default=0,
                        help="scene index (default: 0)")
    parser.add_argument("--max-frames",  type=int,   default=None,
                        help="max frames to process")
    parser.add_argument("--dataroot",    type=str,
                        default="/home/user/AdPerception/data/nuscenes",
                        help="nuScenes dataset root")
    parser.add_argument("--version",     type=str,   default="v1.0-mini",
                        choices=["v1.0-mini", "v1.0-trainval"],
                        help="nuScenes version (default: v1.0-mini)")
    parser.add_argument("--checkpoint",  type=str,
                        default="checkpoints/centerpoint.pth",
                        help="path to trained CenterPoint checkpoint")
    parser.add_argument("--score-thresh", type=float, default=0.05,
                        help="detection score threshold (default: 0.05)")
    args = parser.parse_args()
    main(args)
