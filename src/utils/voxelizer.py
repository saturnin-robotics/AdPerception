"""
    volxelizer.py : Point cloud voxelization for AdPerception pipeline

    Converts a raw lidar point cloud (N, 4+) into a spconv SparseConvtensor ready to be used in PTv3 backbone

    pipeline :
        1. Quantization : (x, y, z) --> (vx, vy, vz) integer voxel indices
        2. Filtering : remove out-of-range points
        3. Encoding : mean-pool all points in each voxel -> one feature vector
        4. Packaging : wrap into spconv.SparseConvTensor
"""

from numpy.ma import count
import torch
import spconv.pytorch as spconv

class Voxelizer:
    """
    Converts a LiDAR point cloud into a spconv SparseConvTensor.
 
    Args:
        voxel_size    : (list[float]) size of each voxel in meters [dx, dy, dz]
        point_range   : (list[float]) [x_min, y_min, z_min, x_max, y_max, z_max]
        max_points    : (int) maximum number of points kept per voxel
        max_voxels    : (int) maximum number of non-empty voxels per frame
 
    Typical nuScenes config:
        voxel_size  = [0.1, 0.1, 0.2]   ->  grid 1024 x 1024 x 40
        point_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    
    """
    def __init__(
            self,
            voxel_size: list,
            point_range: list,
            max_points: int = 10,
            max_voxels: int = 120000):
        self.voxel_size  = torch.tensor(voxel_size, dtype= torch.float32)
        self.point_range = torch.tensor(point_range, dtype= torch.float32)
        self.max_points = max_points
        self.max_voxels = max_voxels

        self.grid_size = torch.round(
            (self.point_range[3:] - self.point_range[:3])/self.voxel_size
        ).long()

    # --------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------


    def __call__(self, points: torch.Tensor, batch_size: int =1, return_inverse: bool= False):

        """
            Args:
                points : (N, 4+) float tensor (x, y, z, intensity, ...)
                for a batch input points[:, 0] contain the batch index and
                points[:, 1:4] contain the points 3D coordinates

                batch_size : number of sample in the batch

            Returns :
                spconv.SparseConvtensor with:
                    features : (V, C) mean-pooled point feature per voxel
                    indices :  (V, 4) [batch_idx, vz, vy, vx] per voxel
                    spatial_shape : [grid_z, grid_y, grid_z]
                    batch_size
        """
        # Single-frame input (no batch size)
        if points.shape[1] == 4:

            batch_idx = torch.zeros((points.shape[0], 1), dtype = torch.float32, device = points.device)
            points = torch.cat([batch_idx, points], dim = 1) # shape of (N, 5)
        
        device = points.device
        self.voxel_size = self.voxel_size.to(device)
        self.point_range = self.point_range.to(device)
        self.grid_size = self.grid_size.to(device)

        # Multi frame input

        # quantization (x, y, z) --> (vx, vy, vz)
        voxel_coords, valid_mask = self._quantize(points)

        # Filtering
        points = points[valid_mask]
        voxel_coords = voxel_coords[valid_mask]

        # Encoding by mean pool (group by voxel and mean-pool features)

        features, sparse_indices, inverse_indices, num_voxels = self._group_and_encode(points, voxel_coords, batch_size)

        # Wrap into SparseConvTensor (expect (Z, Y, X))

        spatial_shape = self.grid_size[[2, 1, 0]].tolist()


        sparse_tensor = spconv.SparseConvTensor(
            features = features,
            indices = sparse_indices,
            spatial_shape = spatial_shape,
            batch_size = batch_size
        )

        if not return_inverse:
            return sparse_tensor
        
        N_original = valid_mask.shape[0]
        valid_indices = torch.where(valid_mask)[0]
        M_kept = inverse_indices.shape[0]

        inverse_indices_full = torch.full(
            (N_original,), fill_value=num_voxels,
            dtype=torch.long, device = device,
        )

        inverse_indices_full[valid_indices[:M_kept]] = inverse_indices

        return sparse_tensor, inverse_indices_full
        
    

    def _quantize(self, points: torch.Tensor):
        """
        Step 1 -- map continuous (x, y, z) to integer voxel indices (vx, vy, vz).
 
        The coordinate origin is shifted so that point_range[:3] maps to
        voxel index (0, 0, 0).
 
        Args:
            points : (N, 5) [batch_idx, x, y, z, intensity]
 
        Returns:
            voxel_coords : (N, 3) int64  [vx, vy, vz]
            valid_mask   : (N,)   bool   True if point is inside the grid
        """
        xyz = points[: ,1:4]
        xyz_shifted = xyz - self.point_range[:3]

        voxel_coords = (xyz_shifted / self.voxel_size).long()

        # Validity 

        valid_mask = (
            (voxel_coords[: ,0] >=0) &
            (voxel_coords[: ,0] < self.grid_size[0])&
            (voxel_coords[: ,1] >=0) &
            (voxel_coords[: ,1] < self.grid_size[1])&
            (voxel_coords[: ,2] >=0) &
            (voxel_coords[: ,2] < self.grid_size[2])
        )
        return voxel_coords, valid_mask

    def _group_and_encode(self, points: torch.Tensor, voxel_coords: torch.Tensor, batch_size: int):
        """
        Steps 2 & 3 -- group points by voxel, then mean-pool their features.
 
        Each unique (batch_idx, vx, vy, vz) tuple defines one non-empty voxel.
        All points in that voxel are averaged into a single feature vector.
 
        We use a hash key trick: encode (batch_idx, vz, vy, vx) as a single
        int64 scalar, run torch.unique to get voxel IDs, then scatter_add for
        the mean -- all GPU-native, no Python loops.
 
        Args:
            points       : (M, 5)  [batch_idx, x, y, z, intensity]
            voxel_coords : (M, 3)  [vx, vy, vz] for each point
            batch_size   : int
 
        Returns:
            features       : (V, C)  mean-pooled features, C = num point channels
            sparse_indices : (V, 4)  [batch_idx, vz, vy, vx] spconv convention
        """

        device = points.device
        Gx = self.grid_size[0]
        Gy = self.grid_size[1]
        Gz = self.grid_size[2]

        batch_idx = points[:, 0].long()

        vx = voxel_coords[:, 0]
        vy = voxel_coords[:, 1]
        vz = voxel_coords[:, 2]

        # Encode four integers into one unique int64 key per voxel

        key = batch_idx * (Gx * Gy * Gz) + vz* (Gx * Gy) + vy * Gx + vx

        # Map each point to its voxel ID (0 ... V-1)

        unique_keys, inverse_indices = torch.unique(key, return_inverse = True)
        num_voxels = unique_keys.shape[0]

        # clip to max_voxels : drop points belonging to voxels beyond the limit
        if num_voxels >= self.max_voxels:
            keep_mask = inverse_indices < self.max_voxels
            points = points[keep_mask]
            inverse_indices = inverse_indices[keep_mask]
            unique_keys = unique_keys[:self.max_voxels]
            num_voxels = self.max_voxels

        # points features to aggregate : [x, y, z, intensity]
        point_features = points[:,1:] # (M, C)
        C = point_features.shape[1]

        # sum features per voxel then mean them

        features_sum = torch.zeros(
            (num_voxels, C), dtype= torch.float32, device= device
        )
        features_sum.scatter_add_(0, inverse_indices.unsqueeze(1).expand(-1, C), point_features,
                                  )
        count = torch.zeros(num_voxels, dtype=torch.float32, device= device)

        count.scatter_add_(0, inverse_indices, torch.ones(points.shape[0], dtype=torch.float32, device = device))

        features = features_sum / count.unsqueeze(1).clamp(min=1.0)

        # decode unique keys back to (batch_id, vx, vy, vz)

        batch_idx_v = unique_keys // (Gx * Gy * Gz)
        remain = unique_keys % (Gx* Gy * Gz)
        vz_v = remain // (Gx * Gy)
        remain = remain % (Gx * Gy)
        vy_v = remain // Gx
        vx_v = remain % Gy

        # spconv convention : indices are (V, 4) [batch_idx, Z, Y, X]

        sparce_indices = torch.stack([batch_idx_v, vz_v, vy_v, vx_v], dim=1).int()
        
        return features,  sparce_indices, inverse_indices, num_voxels

