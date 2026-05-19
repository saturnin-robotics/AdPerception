"""
pipeline.py -- Full LiDAR perception pipeline for AdPerception.

Assembles Voxelizer + PTv3Wrapper + CenterPointHead + SegHead
into a single nn.Module with a clean interface.

Interface:
    input  : points (N, 4)  [x, y, z, intensity]
    output : dict with keys
        "boxes"        : (M, 8)   [cx, cy, cz, l, w, h, sin_t, cos_t]
        "scores"       : (M,)     detection confidence
        "labels"       : (M,)     detection class indices
        "velocity"     : (M, 2)   [vx, vy] for SimpleTrack
        "point_labels" : (N,)     semantic label per input point
        "point_colors" : (N, 3)   RGB color per input point

Tracking (SimpleTrack) is intentionally kept separate in
src/tracking/simpletrack_wrapper.py -- this module outputs
raw detections, the tracker consumes them frame by frame.
"""

from typing import List

import torch
import torch.nn as nn

from src.utils.voxelizer import Voxelizer
from src.models.backbone.ptv3_wrapper import PTv3Wrapper
from src.models.heads.centerpoint import CenterPointHead, NUSCENES_NMS_MIN_DIST
from src.models.heads.seg_head import SegHead


class LiDARPerceptionPipeline(nn.Module):
    """
    End-to-end LiDAR perception pipeline.

    Args:
        voxel_size           : [dx, dy, dz] in meters
        point_range          : [x_min, y_min, z_min, x_max, y_max, z_max]
        max_points_per_voxel : max points kept per voxel before mean-pool
                               (CenterPoint official: 10)
        max_voxels           : maximum non-empty voxels per frame
                               (CenterPoint official train: 120000)
        ptv3_out_ch          : PTv3 output channels fed to both heads
        neck_channels        : CenterPoint neck intermediate channels
        num_det_classes      : detection classes (nuScenes = 10)
        num_seg_classes      : segmentation classes (nuScenes = 17)
        score_thresh         : minimum detection score to keep a box
        max_detections       : maximum boxes per frame
        nms_kernel           : heatmap max-pool kernel size (odd int)
        nms_min_dists        : per-class min BEV center distance for NMS
    """

    def __init__(
        self,
        voxel_size:           list        = [0.1, 0.1, 0.2],
        point_range:          list        = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        max_points_per_voxel: int         = 10,
        max_voxels:           int         = 120_000,
        ptv3_out_ch:          int         = 256,
        neck_channels:        int         = 128,
        num_det_classes:      int         = 10,
        num_seg_classes:      int         = 17,
        score_thresh:         float       = 0.1,
        max_detections:       int         = 500,
        nms_kernel:           int         = 7,
        nms_min_dists:        List[float] = NUSCENES_NMS_MIN_DIST,
    ):
        super().__init__()

        # Grid size [Gz, Gy, Gx] -- spconv convention
        grid_size = [
            round((point_range[5] - point_range[2]) / voxel_size[2]),
            round((point_range[4] - point_range[1]) / voxel_size[1]),
            round((point_range[3] - point_range[0]) / voxel_size[0]),
        ]

        # Voxelizer -- no learnable parameters
        self.voxelizer = Voxelizer(
            voxel_size  = voxel_size,
            point_range = point_range,
            max_points  = max_points_per_voxel,
            max_voxels  = max_voxels,
        )

        # Shared PTv3 backbone --features axtraction
        self.backbone = PTv3Wrapper(
            in_channels  = 4,
            out_channels = ptv3_out_ch,
        )

        # Detection head
        self.det_head = CenterPointHead(
            in_channels    = ptv3_out_ch,
            neck_channels  = neck_channels,
            num_classes    = num_det_classes,
            voxel_size     = voxel_size,
            point_range    = point_range,
            grid_size      = grid_size,
            score_thresh   = score_thresh,
            max_detections = max_detections,
            nms_kernel     = nms_kernel,
            nms_min_dists  = nms_min_dists,
        )

        # Segmentation head -- shares PTv3 features with det_head
        self.seg_head = SegHead(
            in_channels = ptv3_out_ch,
            num_classes = num_seg_classes,
        )

    # =========================================================================
    # Public API
    # =========================================================================

    def forward(
        self,
        points:     torch.Tensor,
        batch_size: int  = 1,
        decode:     bool = True,
    ) -> dict:
        """
        Args:
            points     : (N, 4)  [x, y, z, intensity] on CUDA
            batch_size : number of samples (1 for single-frame inference)
            decode     : if True return decoded 3D boxes (inference)
                         if False return raw head logits (training)

        Returns:
            dict -- see module docstring for output keys
        """
        N = points.shape[0]

        # Step 1 -- voxelization + inverse_indices for seg propagation
      
        sparse, inverse_indices = self.voxelizer(   # type: ignore[misc]
            points,
            batch_size = batch_size,
            return_inverse = True

        )

        # Step 2 -- PTv3 backbone -- features shared between both heads
        features = self.backbone(sparse)

        # Step 3 -- detection head
        det_out = self.det_head(
            features,
            sparse.indices,
            batch_size = batch_size,
            decode     = decode,
        )

        # Step 4 -- segmentation head
        seg_out = self.seg_head(
            features,
            inverse_indices = inverse_indices,
            num_points      = N,
        )

        # Assemble output
        if decode:
            dets = det_out["detections"][0]    # batch index 0
            return {
                "boxes"        : dets["boxes"],
                "scores"       : dets["scores"],
                "labels"       : dets["labels"],
                "velocity"     : dets["velocity"],
                "point_labels" : seg_out["point_labels"],
                "point_colors" : seg_out["point_colors"],
            }
        else:
            return {
                "det_preds" : det_out,
                "seg_preds" : seg_out,
            }

    # =========================================================================
    # Checkpoint utilities
    # =========================================================================

    def load_sonata(self, checkpoint_path: str) -> None:
        """
        Load Sonata pretrained weights into the PTv3 backbone.

        strict=False allows loading backbone weights while ignoring
        Sonata's segmentation head weights that differ from our SegHead.

        Args:
            checkpoint_path : path to sonata.pth
        """
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        state_dict = checkpoint.get(
            "state_dict", checkpoint.get("model", checkpoint)
        )

        # Filter out keys whose shape does not match the current model
        # e.g. embedding.stem.linear.weight: [48, 9] vs [48, 4] (in_channels diff)
        model_sd   = self.backbone.backbone.state_dict()
        compatible = {
            k: v for k, v in state_dict.items()
            if k in model_sd and v.shape == model_sd[k].shape
        }
        skipped = [k for k in state_dict if k not in compatible]

        missing, _ = self.backbone.backbone.load_state_dict(
            compatible, strict=False
        )
        print(
            f"Sonata -- loaded: {len(compatible)}, "
            f"missing: {len(missing)}, "
            f"skipped (shape mismatch): {len(skipped)}"
        )
        if skipped:
            print(
                f"  skipped keys: {skipped[:5]}"
                f"{'...' if len(skipped) > 5 else ''}"
            )
