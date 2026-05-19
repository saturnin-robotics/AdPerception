"""
infer.py -- AdPerception inference + Rerun visualization.

Usage:
    python infer.py                       # GT boxes + seg colors    (~110 ms/frame)
    python infer.py --use-gt              # GT boxes + height colors (~2 ms/frame)
    python infer.py --use-gt --scene 2    # scene 2 (40 frames)
    python train.py                        # train CenterPoint head (~15 min)
    python infer.py --predicted           # trained head + GT overlay (~110 ms/frame)
    python infer.py --max-frames 50

Default mode (no flags): runs PTv3 backbone for semantic seg colors, uses
nuScenes GT boxes for detection. The CenterPoint head is untrained --
use --predicted only to inspect raw head output (expect noisy/biased results).
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

# dynamic classes to track (exclude barrier=5, traffic_cone=9)
DYNAMIC_CLASSES = {1, 2, 3, 4, 6, 7, 8}


# Helpers

def _height_colors(pts_np: np.ndarray) -> np.ndarray:
    """
    Colors LiDAR points by Z height -- rainbow gradient, -2m (blue) to +4m (red).
    Pure numpy, ~0.5 ms for 35k points.
    """
    t = np.clip((pts_np[:, 2] + 2.0) / 6.0, 0.0, 1.0)
    stops = [0.0, 0.25, 0.5, 0.75, 1.0]
    r = np.interp(t, stops, [  0,   0,   0, 255, 255])
    g = np.interp(t, stops, [  0, 255, 255, 255,   0])
    b = np.interp(t, stops, [255, 255,   0,   0,   0])
    return np.stack([r, g, b], axis=1).astype(np.uint8)


def _gt_out(sample: dict, device: torch.device) -> dict:
    """
    Builds a pipeline-compatible output dict from GT nuScenes annotations.
    No GPU required -- runs in < 0.2 ms.

    GT box format    : (M, 7)  [cx, cy, cz, l, w, h, yaw]
    Pipeline box fmt : (M, 8)  [cx, cy, cz, l, w, h, sin_t, cos_t]
    """
    gt   = sample["gt_boxes"]   # (M, 7)  CPU tensor
    lbls = sample["gt_labels"]  # (M,)    CPU tensor
    yaw  = gt[:, 6]
    boxes_8 = torch.cat([gt[:, :6], torch.sin(yaw).unsqueeze(1),
                                    torch.cos(yaw).unsqueeze(1)], dim=1)
    pts_np    = sample["points"].numpy()
    pt_colors = torch.from_numpy(_height_colors(pts_np))

    return {
        "boxes"        : boxes_8.to(device),
        "scores"       : torch.ones(len(gt), device=device),
        "labels"       : lbls.to(device),
        "velocity"     : torch.zeros(len(gt), 2, device=device),
        "point_colors" : pt_colors,          # CPU tensor -- viz does .cpu() anyway
    }


# main 

def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda")

    dataset = NuScenesLiDARDataset(
        dataroot = "/home/user/AdPerception/data/nuscenes",
        version  = "v1.0-mini",
        verbose  = False,
    )

    # pipeline used for semantic seg colors (backbone pretrained via Sonata)
    # detection always comes from GT unless --predicted is set
    pipeline = None
    if not args.use_gt:
        pipeline = LiDARPerceptionPipeline(
            voxel_size     = [0.2, 0.2, 0.4],
            score_thresh   = 0.15,
            max_detections = 40,
            nms_kernel     = 7,
        ).to(device)
        pipeline.load_sonata("checkpoints/sonata.pth")

        if args.predicted:
            ckpt_path = "checkpoints/centerpoint.pth"
            if os.path.exists(ckpt_path):
                state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                pipeline.load_state_dict(state["state_dict"], strict=False)
                print(f"Loaded trained CenterPoint checkpoint: {ckpt_path}")
            else:
                print(
                    f"[warn] No trained checkpoint at {ckpt_path}. "
                    "Run 'python train.py' first for meaningful detections."
                )

        pipeline.eval()
        print("Warming up CUDA kernels...")
        with torch.no_grad(), torch.cuda.amp.autocast():
            _ = pipeline(dataset[0]["points"].to(device))
        print("Done.")

    # read LiDAR calibration once -- gives the sensor's position in ego frame.
    # The x-component (≈ 0.944 m forward of the rear axle) is used both to
    # position the ego box correctly in the viewer and to suppress spurious
    # detections whose centre falls inside the ego vehicle's footprint.
    _sample0    = dataset.nusc.get("sample", dataset.sample_tokens[0])
    _sd0        = dataset.nusc.get("sample_data", _sample0["data"]["LIDAR_TOP"])
    _calib0     = dataset.nusc.get("calibrated_sensor", _sd0["calibrated_sensor_token"])
    ego_cx      = float(_calib0["translation"][0])   # ≈ 0.944 m
    # ego footprint half-extents (Renault Zoe: 4.08 m × 1.77 m)
    EGO_HALF_L  = 2.04   # half-length along X
    EGO_HALF_W  = 0.89   # half-width  along Y

    tracker = SimpleTrackWrapper(num_classes=10, max_trail_len=10)
    viz     = Visualizer(app_name="AdPerception", ego_cx=ego_cx)

    start_idx = args.scene * 10
    end_idx   = min(
        start_idx + args.max_frames, len(dataset)
    ) if args.max_frames else len(dataset)

    trained_ckpt = os.path.exists("checkpoints/centerpoint.pth")
    if args.use_gt:
        mode = "GT boxes + height colors  (no pipeline)"
    elif args.predicted:
        mode = (
            "predicted boxes + seg colors  (trained CenterPoint)"
            if trained_ckpt else
            "predicted boxes + seg colors  (untrained head -- run train.py first)"
        )
    else:
        mode = "GT boxes + seg colors  (PTv3 Sonata backbone)"
    print(f"\nMode   : {mode}")
    print(f"Frames : {start_idx} → {end_idx - 1}  ({end_idx - start_idx} frames)\n")
    print(f"{'Frame':>5} | {'Pts':>6} | {'Dets':>4} | {'Trk':>4} | {'Percep':>8} | {'Viz':>6}")
    print("─" * 52)

    prev_scene_token = None

    try:
        for i in range(start_idx, end_idx):
            sample = dataset[i]

            # reset tracker at scene boundary
            scene_token = dataset.nusc.get("sample", sample["sample_token"])["scene_token"]
            if scene_token != prev_scene_token:
                if prev_scene_token is not None:
                    print("**************************\nnew scene -- tracker reset\n**************************")
                tracker.reset()
                prev_scene_token = scene_token

            #  perception 
            t0 = time.perf_counter()

            if args.use_gt:
                # pure GT: no pipeline, height colors
                out     = _gt_out(sample, device)
                pts_gpu = sample["points"]          # stays on CPU, viz handles it
            elif args.predicted:
                # untrained head -- expect noisy/biased detections
                pts_gpu = sample["points"].to(device)
                with torch.no_grad(), torch.cuda.amp.autocast():
                    out = pipeline(pts_gpu)         # type: ignore[union-attr]
            else:
                # default: GT boxes + PTv3 Sonata seg colors
                pts_gpu  = sample["points"].to(device)
                with torch.no_grad(), torch.cuda.amp.autocast():
                    seg_out = pipeline(pts_gpu)     # type: ignore[union-attr]
                gt_part  = _gt_out(sample, device)
                out = {
                    "boxes"        : gt_part["boxes"],
                    "scores"       : gt_part["scores"],
                    "labels"       : gt_part["labels"],
                    "velocity"     : gt_part["velocity"],
                    "point_colors" : seg_out["point_colors"],   # from Sonata backbone
                    "point_labels" : seg_out["point_labels"],   # for driveable zone overlay
                }

            t1 = time.perf_counter()

            #  filter to dynamic classes
            labels_np = out["labels"].cpu().numpy()
            dyn_mask  = np.isin(labels_np, list(DYNAMIC_CLASSES))

            boxes_dyn    = out["boxes"][dyn_mask]
            scores_dyn   = out["scores"][dyn_mask]
            labels_dyn   = out["labels"][dyn_mask]
            velocity_dyn = out["velocity"][dyn_mask]

            # suppress detections whose centre falls inside the ego footprint.
            # Necessary in predicted mode (untrained head generates random boxes,
            # some of which land on the ego vehicle's LiDAR self-returns).
            # Has no effect in GT mode (nuScenes GT never annotates the ego itself).
            boxes_xy    = boxes_dyn.cpu()
            not_in_ego  = ~(
                ((boxes_xy[:, 0] - ego_cx).abs() < EGO_HALF_L) &
                (boxes_xy[:, 1].abs()             < EGO_HALF_W)
            )
            boxes_dyn    = boxes_dyn[not_in_ego]
            scores_dyn   = scores_dyn[not_in_ego]
            labels_dyn   = labels_dyn[not_in_ego]
            velocity_dyn = velocity_dyn[not_in_ego]

            #  tracking
            tracks = tracker.update(
                boxes    = boxes_dyn,
                scores   = scores_dyn,
                labels   = labels_dyn,
                velocity = velocity_dyn,
            )

            # visualization
            t2 = time.perf_counter()
            # GT overlay only in --predicted mode (comparison predicted vs GT).
            # In default/GT-only modes the tracks already ARE the GT -- no overlay needed.
            show_gt = args.predicted
            viz.update(
                pts_gpu, out, tracks=tracks, frame_idx=i,
                gt_boxes  = sample["gt_boxes"]  if show_gt else None,
                gt_labels = sample["gt_labels"] if show_gt else None,
            )
            t3 = time.perf_counter()

            print(
                f"{i:>5} | {sample['points'].shape[0]:>6} | "
                f"{int(dyn_mask.sum()):>4} | {len(tracks):>4} | "
                f"{1000*(t1-t0):>6.0f}ms | {1000*(t3-t2):>4.0f}ms"
            )

    except KeyboardInterrupt:
        print("\nStopped -- Rerun viewer stays open for review")
    finally:
        viz.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-gt",    action="store_true",
                        help="GT boxes + height colors (no GPU, ~2 ms/frame)")
    parser.add_argument("--predicted", action="store_true",
                        help="untrained CenterPoint head -- expect noisy results")
    parser.add_argument("--scene",     type=int, default=0,
                        help="scene index 0-9")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()
    main(args)
