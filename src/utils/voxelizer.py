"""
voxelizer.py -- Point cloud voxelization for AdPerception pipeline.

Converts a raw LiDAR point cloud (N, 4+) into a spconv SparseConvTensor
ready to be fed into the PTv3 backbone.

Pipeline:
    1. Quantization  : (x, y, z) -> (vx, vy, vz) integer voxel indices
    2. Filtering     : remove out-of-range points
    3. Clipping      : keep at most max_points points per voxel
    4. Encoding      : mean-pool remaining points per voxel
    5. Packaging     : wrap into spconv.SparseConvTensor
"""

import torch
import spconv.pytorch as spconv
from typing import Union, Tuple


class Voxelizer:
    """
    Converts a LiDAR point cloud into a spconv SparseConvTensor.

    Args:
        voxel_size    : [dx, dy, dz] in meters
        point_range   : [x_min, y_min, z_min, x_max, y_max, z_max]
        max_points    : maximum number of points kept per voxel
                        (CenterPoint official: 10)
        max_voxels    : maximum number of non-empty voxels per frame
                        (CenterPoint official train: 120000, test: 160000)

    Official CenterPoint nuScenes config (0.075 m):
        voxel_size  = [0.075, 0.075, 0.2]   -> grid 1440 x 1440 x 40
        point_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
        max_points  = 10
        max_voxels  = 120000 (train) / 160000 (test)

    Local debug config (0.1 m):
        voxel_size  = [0.1, 0.1, 0.2]       -> grid 1024 x 1024 x 40
        point_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        max_points  = 10
        max_voxels  = 80000
    """

    def __init__(
        self,
        voxel_size:  list,
        point_range: list,
        max_points:  int = 10,
        max_voxels:  int = 120_000,
    ):
        self.voxel_size  = torch.tensor(voxel_size,  dtype=torch.float32)
        self.point_range = torch.tensor(point_range, dtype=torch.float32)
        self.max_points  = max_points
        self.max_voxels  = max_voxels

        self.grid_size = torch.round(
            (self.point_range[3:] - self.point_range[:3]) / self.voxel_size
        ).long()

    # =========================================================================
    # Public API
    # =========================================================================

    def __call__(
        self,
        points:         torch.Tensor,
        batch_size:     int  = 1,
        return_inverse: bool = False,
    )->Union[spconv.SparseConvTensor, Tuple[spconv.SparseConvTensor, torch.Tensor]]:
        """
        Args:
            points         : (N, 4) float tensor [x, y, z, intensity]
                             Single-frame input -- batch index added internally.
            batch_size     : number of samples in the batch (1 for inference)
            return_inverse : if True, also return full-size inverse_indices
                             mapping each original point to its voxel index.

        Returns:
            sparse_tensor        : spconv.SparseConvTensor
            inverse_indices_full : (N,) long -- only when return_inverse=True.
                                   Out-of-range or dropped points get sentinel
                                   value = num_voxels.
        """
        # Add batch dimension for single-frame input (N, 4) -> (N, 5)
        if points.shape[1] == 4:
            batch_idx = torch.zeros(
                (points.shape[0], 1), dtype=torch.float32, device=points.device
            )
            points = torch.cat([batch_idx, points], dim=1)

        device = points.device
        self.voxel_size  = self.voxel_size.to(device)
        self.point_range = self.point_range.to(device)
        self.grid_size   = self.grid_size.to(device)

        # Step 1 -- quantization
        voxel_coords, valid_mask = self._quantize(points)

        # Step 2 -- filter out-of-range points
        points       = points[valid_mask]
        voxel_coords = voxel_coords[valid_mask]

        # Steps 3 + 4 -- clip per-voxel + mean-pool
        features, sparse_indices, inverse_indices, num_voxels, kept_mask = \
            self._group_and_encode(points, voxel_coords, batch_size)

        # Wrap into SparseConvTensor -- spconv expects (Z, Y, X) spatial shape
        spatial_shape = self.grid_size[[2, 1, 0]].tolist()
        sparse_tensor = spconv.SparseConvTensor(
            features      = features,
            indices       = sparse_indices,
            spatial_shape = spatial_shape,
            batch_size    = batch_size,
        )

        if not return_inverse:
            return sparse_tensor

        # Build full-size inverse_indices (one entry per original point).
        # Points outside point_range or dropped by max_voxels get
        # fill_value = num_voxels (sentinel used by SegHead._propagate_to_points).
        N_original         = valid_mask.shape[0]
        valid_indices      = torch.where(valid_mask)[0]
        kept_valid_indices = valid_indices[kept_mask]

        inverse_indices_full = torch.full(
            (N_original,), fill_value=num_voxels,
            dtype=torch.long, device=device,
        )
        inverse_indices_full[kept_valid_indices] = inverse_indices

        return sparse_tensor, inverse_indices_full

    # =========================================================================
    # Private helpers
    # =========================================================================

    def _quantize(self, points: torch.Tensor):
        """
        Map continuous (x, y, z) to integer voxel indices (vx, vy, vz).

        The origin is shifted so that point_range[:3] maps to voxel (0, 0, 0).

        Args:
            points : (N, 5) [batch_idx, x, y, z, intensity]

        Returns:
            voxel_coords : (N, 3) int64  [vx, vy, vz]
            valid_mask   : (N,)   bool   True if point is inside the grid
        """
        xyz          = points[:, 1:4]
        xyz_shifted  = xyz - self.point_range[:3]
        voxel_coords = (xyz_shifted / self.voxel_size).long()

        valid_mask = (
            (voxel_coords[:, 0] >= 0) & (voxel_coords[:, 0] < self.grid_size[0]) &
            (voxel_coords[:, 1] >= 0) & (voxel_coords[:, 1] < self.grid_size[1]) &
            (voxel_coords[:, 2] >= 0) & (voxel_coords[:, 2] < self.grid_size[2])
        )
        return voxel_coords, valid_mask

    def _group_and_encode(
        self,
        points:       torch.Tensor,
        voxel_coords: torch.Tensor,
        batch_size:   int,
    ):
        """
        Group points by voxel, clip to max_points per voxel, then mean-pool.

        Uses a hash-key trick: encode (batch_idx, vz, vy, vx) as a single
        int64 key, run torch.unique to get voxel IDs, scatter_add for mean.
        All operations are GPU-native -- no Python loops.

        Args:
            points       : (M, 5)  [batch_idx, x, y, z, intensity]
            voxel_coords : (M, 3)  [vx, vy, vz]
            batch_size   : int

        Returns:
            features        : (V, C)    mean-pooled features
            sparse_indices  : (V, 4)    [batch_idx, vz, vy, vx] spconv convention
            inverse_indices : (M_kept,) voxel index for each kept point
            num_voxels      : int       number of non-empty voxels after clip
            kept_mask       : (M,) bool points that survived both clips
        """
        device = points.device
        Gx = self.grid_size[0]
        Gy = self.grid_size[1]
        Gz = self.grid_size[2]

        batch_idx = points[:, 0].long()
        vx = voxel_coords[:, 0]
        vy = voxel_coords[:, 1]
        vz = voxel_coords[:, 2]

        # Encode (batch_idx, vz, vy, vx) -> single int64 key per voxel
        key = batch_idx * (Gx * Gy * Gz) + vz * (Gx * Gy) + vy * Gx + vx

        # Map each point to its voxel ID in [0, num_voxels)
        unique_keys, inverse_indices = torch.unique(key, return_inverse=True)
        num_voxels = unique_keys.shape[0]

        # =====================================================================
        # Per-voxel max_points clipping
        # Compute a local index for each point within its voxel (0, 1, 2 ...)
        # Keep only points with local_idx < max_points.
        # =====================================================================
        order      = torch.argsort(inverse_indices, stable=True)
        inv_sorted = inverse_indices[order]

        _, counts = torch.unique_consecutive(inv_sorted, return_counts=True)

        # Build local per-voxel indices: voxel with k points gets 0, 1, ..., k-1
        local_idx_sorted = torch.cat([
            torch.arange(c.item(), dtype=torch.long, device=device)
            for c in counts
        ])
        local_idx = torch.empty_like(local_idx_sorted)
        local_idx[order] = local_idx_sorted

        per_voxel_mask = local_idx < self.max_points

        # =====================================================================
        # max_voxels clipping
        # Drop points belonging to voxels beyond the limit.
        # =====================================================================
        max_voxels_mask = inverse_indices < self.max_voxels

        # Combined mask: point must survive both clips
        kept_mask = per_voxel_mask & max_voxels_mask

        points          = points[kept_mask]
        inverse_indices = inverse_indices[kept_mask]

        if num_voxels > self.max_voxels:
            unique_keys = unique_keys[:self.max_voxels]
            num_voxels  = self.max_voxels

        # =====================================================================
        # Mean-pool features
        # =====================================================================
        point_features = points[:, 1:]    # (M_kept, C) -- drop batch_idx column
        C = point_features.shape[1]

        features_sum = torch.zeros(
            (num_voxels, C), dtype=torch.float32, device=device
        )
        features_sum.scatter_add_(
            0,
            inverse_indices.unsqueeze(1).expand(-1, C),
            point_features,
        )

        count = torch.zeros(num_voxels, dtype=torch.float32, device=device)
        count.scatter_add_(
            0,
            inverse_indices,
            torch.ones(points.shape[0], dtype=torch.float32, device=device),
        )

        features = features_sum / count.unsqueeze(1).clamp(min=1.0)

        # =====================================================================
        # Decode keys back to (batch_idx, vz, vy, vx)
        # =====================================================================
        batch_idx_v = unique_keys // (Gx * Gy * Gz)
        remain      = unique_keys % (Gx * Gy * Gz)
        vz_v        = remain // (Gx * Gy)
        remain      = remain % (Gx * Gy)
        vy_v        = remain // Gx
        vx_v        = remain % Gx

        # spconv indices: (V, 4) [batch_idx, Z, Y, X]
        sparse_indices = torch.stack(
            [batch_idx_v, vz_v, vy_v, vx_v], dim=1
        ).int()

        return features, sparse_indices, inverse_indices, num_voxels, kept_mask
