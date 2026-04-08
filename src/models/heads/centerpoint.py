"""
centerpoint.py - CenterPoint detection head for AdPerception.

Takes PTv3 voxel features (V, C) in sparse 3D space and produces
3D bounding boxes via a BEV heatmap + regression pipeline.

Pipeline:
    1. BEV collapse   : sparse (V, C) -> dense (B, C, H, W) via max-pool
    2. Shared neck    : 2D conv blocks to enrich BEV features
    3. Six heads      : heatmap, offset, height, dims, rotation, velocity
    4. Decode         : peak extraction + regression -> 3D boxes

Output per detection:
    (cx, cy, cz, l, w, h, sin_theta, cos_theta, vx, vy, score, class_id)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv


class BEVCollapse(nn.Module):
    """
    Converts sparse 3D voxel features to a dense 2D BEV feature map.

    For each (batch, vx, vy) column, keeps the max over all vz levels.
    Max pool preserves the most salient features - stronger signal than
    mean pool for detection.

    Does NOT materialize the full 3D volume (B, C, Gz, Gy, Gx) which
    would require ~40 GB for nuScenes resolution. Instead scatters
    directly into a 2D BEV grid, writing voxels in ascending feature-norm
    order so the highest-norm voxel wins each (b, y, x) cell.

    Args:
        in_channels : number of input feature channels from PTv3
        grid_size   : [Gz, Gy, Gx] spatial shape of the sparse tensor
    """

    def __init__(self, in_channels: int, grid_size: list):
        super().__init__()
        self.in_channels = in_channels
        self.Gz = grid_size[0]
        self.Gy = grid_size[1]
        self.Gx = grid_size[2]

    def forward(
        self,
        features:   torch.Tensor,
        indices:    torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """
        Args:
            features   : (V, C)  sparse voxel features from PTv3
            indices    : (V, 4)  [batch, Z, Y, X] spconv indices
            batch_size : int

        Returns:
            bev : (B, C, Gy, Gx)  dense BEV feature map
        """
        device = features.device
        C = features.shape[1]

        b = indices[:, 0].long()  # (V,)
        y = indices[:, 2].long()  # (V,)
        x = indices[:, 3].long()  # (V,)

        # sort by feature norm ascending so highest-norm voxel wins each cell
        norms = features.norm(dim=1)
        order = norms.argsort()
        features = features[order]
        b = b[order]
        y = y[order]
        x = x[order]

        # encode (b, y, x) as a single flat index into (B, Gy, Gx)
        flat_idx = b * (self.Gy * self.Gx) + y * self.Gx + x  # (V,)

        # allocate flat BEV (B*Gy*Gx, C) and scatter features
        bev_flat = torch.zeros(
            (batch_size * self.Gy * self.Gx, C),
            dtype  = features.dtype,
            device = device,
        )

        # expand flat_idx to (V, C) for scatter
        flat_idx_expanded = flat_idx.unsqueeze(1).expand(-1, C)  # (V, C)

        # scatter - last write wins (highest norm written last)
        bev_flat.scatter_(0, flat_idx_expanded, features)

        # reshape to (B, Gy, Gx, C) then permute to (B, C, Gy, Gx)
        bev = bev_flat.view(batch_size, self.Gy, self.Gx, C)
        bev = bev.permute(0, 3, 1, 2).contiguous()  # (B, C, Gy, Gx)

        return bev


class SharedNeck(nn.Module):
    """
    2D convolutional neck applied on the BEV feature map.

    Enriches spatial context after BEV collapse before the detection heads.
    Two conv blocks with BatchNorm and ReLU - lightweight by design since
    PTv3 already produces rich features.

    padding=1 on kernel_size=3 preserves H x W resolution exactly.

    Args:
        in_channels  : C from BEV collapse output
        out_channels : channels consumed by all detection heads
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.neck = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, bev: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bev : (B, in_channels, H, W)

        Returns:
            (B, out_channels, H, W)
        """
        return self.neck(bev)


class DetectionHeads(nn.Module):
    """
    Six independent 1x1 conv heads applied on shared BEV features.

    Each head is a single Conv2d - no shared parameters between heads.
    1x1 conv = per-cell MLP - predicts one value per BEV cell.

    Heads:
        heatmap  : (B, num_classes, H, W)  Gaussian peak per object center
        offset   : (B, 2, H, W)            sub-voxel xy offset (dx, dy)
        height   : (B, 1, H, W)            z coordinate of object center
        dims     : (B, 3, H, W)            log(l, w, h) object dimensions
        rotation : (B, 2, H, W)            (sin theta, cos theta) heading
        velocity : (B, 2, H, W)            (vx, vy) for SimpleTrack MOT

    Args:
        in_channels  : channels from SharedNeck output
        num_classes  : number of object classes (nuScenes = 10)
    """

    def __init__(self, in_channels: int, num_classes: int = 10):
        super().__init__()

        self.heatmap  = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        self.offset   = nn.Conv2d(in_channels, 2,           kernel_size=1)
        self.height   = nn.Conv2d(in_channels, 1,           kernel_size=1)
        self.dims     = nn.Conv2d(in_channels, 3,           kernel_size=1)
        self.rotation = nn.Conv2d(in_channels, 2,           kernel_size=1)
        self.velocity = nn.Conv2d(in_channels, 2,           kernel_size=1)

    def forward(self, bev: torch.Tensor) -> dict:
        """
        Args:
            bev : (B, in_channels, H, W)

        Returns:
            dict with keys:
                "heatmap"  : (B, num_classes, H, W)  raw logits
                "offset"   : (B, 2, H, W)
                "height"   : (B, 1, H, W)
                "dims"     : (B, 3, H, W)             log scale
                "rotation" : (B, 2, H, W)             sin/cos
                "velocity" : (B, 2, H, W)
        """
        return {
            "heatmap"  : self.heatmap(bev),
            "offset"   : self.offset(bev),
            "height"   : self.height(bev),
            "dims"     : self.dims(bev),
            "rotation" : self.rotation(bev),
            "velocity" : self.velocity(bev),
        }


class BoxDecoder:
    """
    Decodes raw head outputs into 3D bounding boxes.

    Steps:
        1. Peak extraction : find local maxima in heatmap (one per object)
        2. Gather          : collect regression values at peak locations
        3. Reconstruct     : convert offsets/log-dims/sin-cos to real values

    Not an nn.Module - no learnable parameters, pure geometry.
    """

    def __init__(
        self,
        voxel_size:     list,
        point_range:    list,
        score_thresh:   float = 0.1,
        max_detections: int   = 500,
        nms_kernel:     int   = 3,
    ):
        self.voxel_size     = voxel_size
        self.point_range    = point_range
        self.score_thresh   = score_thresh
        self.max_detections = max_detections
        self.nms_kernel     = nms_kernel

        self.dx    = voxel_size[0]
        self.dy    = voxel_size[1]
        self.x_min = point_range[0]
        self.y_min = point_range[1]

    def decode(self, preds: dict, batch_size: int) -> list:
        """
        Args:
            preds      : dict from DetectionHeads.forward()
            batch_size : int

        Returns:
            list of length batch_size, each element is a dict:
                "boxes"    : (N, 8)  [cx, cy, cz, l, w, h, sin_t, cos_t]
                "scores"   : (N,)    confidence scores
                "labels"   : (N,)    class indices
                "velocity" : (N, 2)  [vx, vy] for SimpleTrack
        """
        heatmap  = torch.sigmoid(preds["heatmap"])
        offset   = preds["offset"]
        height   = preds["height"]
        dims     = preds["dims"]
        rotation = preds["rotation"]
        velocity = preds["velocity"]

        results = []
        for b in range(batch_size):
            boxes, scores, labels, vels = self._decode_single(
                heatmap[b], offset[b], height[b],
                dims[b], rotation[b], velocity[b],
            )
            results.append({
                "boxes"    : boxes,
                "scores"   : scores,
                "labels"   : labels,
                "velocity" : vels,
            })

        return results

    def _decode_single(
        self,
        heatmap:  torch.Tensor,  # (K, H, W)
        offset:   torch.Tensor,  # (2, H, W)
        height:   torch.Tensor,  # (1, H, W)
        dims:     torch.Tensor,  # (3, H, W)
        rotation: torch.Tensor,  # (2, H, W)
        velocity: torch.Tensor,  # (2, H, W)
    ):
        K, H, W = heatmap.shape
        device  = heatmap.device

        # step 1 - peak extraction via max pooling (replaces NMS)
        heatmap_max = F.max_pool2d(
            heatmap, kernel_size=self.nms_kernel,
            stride=1, padding=self.nms_kernel // 2,
        )
        peak_mask = (heatmap == heatmap_max) & (heatmap > self.score_thresh)

        # step 2 - gather peak locations and scores
        class_ids, ys, xs = peak_mask.nonzero(as_tuple=True)
        scores = heatmap[class_ids, ys, xs]

        # keep top max_detections by score
        if scores.shape[0] > self.max_detections:
            topk_scores, topk_idx = scores.topk(self.max_detections)
            scores    = topk_scores
            class_ids = class_ids[topk_idx]
            ys        = ys[topk_idx]
            xs        = xs[topk_idx]

        P = scores.shape[0]

        if P == 0:
            empty = torch.zeros((0, 8), device=device)
            return empty, scores, class_ids, torch.zeros((0, 2), device=device)

        # step 3 - reconstruct box coordinates from regression outputs
        dx = offset[0, ys, xs]
        dy = offset[1, ys, xs]

        # continuous x, y in meters from voxel grid indices
        cx = (xs.float() + dx) * self.dx + self.x_min + self.dx / 2
        cy = (ys.float() + dy) * self.dy + self.y_min + self.dy / 2
        cz = height[0, ys, xs]

        # dimensions: exp to recover real values from log predictions
        l = dims[0, ys, xs].exp()
        w = dims[1, ys, xs].exp()
        h = dims[2, ys, xs].exp()

        sin_t = rotation[0, ys, xs]
        cos_t = rotation[1, ys, xs]

        vx = velocity[0, ys, xs]
        vy = velocity[1, ys, xs]

        boxes = torch.stack([cx, cy, cz, l, w, h, sin_t, cos_t], dim=1)
        vels  = torch.stack([vx, vy], dim=1)

        return boxes, scores, class_ids, vels


class CenterPointHead(nn.Module):
    """
    Full CenterPoint detection head for AdPerception.

    Assembles BEVCollapse + SharedNeck + DetectionHeads + BoxDecoder
    into a single nn.Module consumed by pipeline.py.

    Args:
        in_channels     : PTv3 output channels (default 256)
        neck_channels   : intermediate BEV channels in SharedNeck
        num_classes     : number of object classes (nuScenes = 10)
        voxel_size      : [dx, dy, dz] in meters
        point_range     : [x_min, y_min, z_min, x_max, y_max, z_max]
        grid_size       : [Gz, Gy, Gx] from voxelizer
        score_thresh    : minimum heatmap score to keep a detection
        max_detections  : maximum number of boxes per frame
    """

    def __init__(
        self,
        in_channels:    int   = 256,
        neck_channels:  int   = 128,
        num_classes:    int   = 10,
        voxel_size:     list  = [0.1, 0.1, 0.2],
        point_range:    list  = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        grid_size:      list  = [40, 1024, 1024],
        score_thresh:   float = 0.1,
        max_detections: int   = 500,
    ):
        super().__init__()

        self.collapse = BEVCollapse(in_channels, grid_size)
        self.neck     = SharedNeck(in_channels, neck_channels)
        self.heads    = DetectionHeads(neck_channels, num_classes)
        self.decoder  = BoxDecoder(
            voxel_size     = voxel_size,
            point_range    = point_range,
            score_thresh   = score_thresh,
            max_detections = max_detections,
        )

    def forward(
        self,
        features:   torch.Tensor,
        indices:    torch.Tensor,
        batch_size: int,
        decode:     bool = False,
    ) -> dict:
        """
        Args:
            features   : (V, C)   PTv3 voxel features
            indices    : (V, 4)   [batch, Z, Y, X] spconv indices
            batch_size : int
            decode     : if True run BoxDecoder and return 3D boxes
                         if False return raw head predictions for training

        Returns:
            if decode=False:
                dict with raw head tensors (heatmap, offset, height,
                dims, rotation, velocity)
            if decode=True:
                dict with key "detections" : list of per-sample dicts
                    each containing boxes, scores, labels, velocity
        """
        # 1 - sparse 3D -> dense BEV
        bev = self.collapse(features, indices, batch_size)  # (B, C, Gy, Gx)

        # 2 - enrich BEV features
        bev = self.neck(bev)                                # (B, neck_C, Gy, Gx)

        # 3 - six regression heads
        preds = self.heads(bev)                             # dict of tensors

        if not decode:
            return preds

        # 4 - decode to 3D boxes (inference only)
        detections = self.decoder.decode(preds, batch_size)
        return {"detections": detections}