"""
seg_head.py - Semantic segmentation head for AdPerception.

Takes PTv3 voxel features (V, C) and predicts a semantic class
label per voxel, then propagates labels back to original points.

Pipeline:
    1. MLP       : (V, C) -> (V, num_classes)  per-voxel class scores
    2. Propagate : voxel labels -> point labels via voxel membership

nuScenes LiDAR segmentation classes (17 total):
    0  noise           1  barrier          2  bicycle
    3  bus             4  car              5  construction_vehicle
    6  motorcycle      7  pedestrian       8  traffic_cone
    9  trailer         10 truck            11 driveable_surface
    12 other_flat      13 sidewalk         14 terrain
    15 manmade         16 vegetation
"""



from typing import Optional

import torch
import torch.nn as nn

class SegHead(nn.Module):
    """
    Semantic segmentation head for AdPerception.

    A lightweight two-layer MLP applied per voxel independently.
    PTv3 already captures spatial context via attention -- the MLP
    only needs to map enriched features to class scores.

    Args:
        in_channels  : PTv3 output channels (default 256)
        mid_channels : intermediate MLP channels (default 128)
        num_classes  : number of semantic classes (nuScenes = 17)
        dropout      : dropout rate between MLP layers (default 0.1)
    """

    # nuscenes LiDAR segmentation class names
    CLASS_NAMES = [
        "noise", "barrier", "bicycle", "bus", "car", "construction_vehicle", "motorcycle",
        "pedestrian", "traffic_cone", "trailer", "truck", "driveable_surface", "other_flat",
        "sidewalk", "terrain", "manmade", "vegetation"
    ]
    
    CLASS_COLORS = torch.tensor([
        [0,   0,   0  ],  # noise               - black
        [255, 120, 50 ],  # barrier             - orange
        [255, 192, 203],  # bicycle             - pink
        [255, 255, 0  ],  # bus                 - yellow
        [0,   150, 245],  # car                 - blue
        [0,   255, 255],  # construction_vehicle- cyan
        [200, 180, 0  ],  # motorcycle          - dark yellow
        [255, 0,   0  ],  # pedestrian          - red
        [255, 240, 150],  # traffic_cone        - light yellow
        [135, 60,  0  ],  # trailer             - brown
        [160, 32,  240],  # truck               - purple
        [255, 0,   255],  # driveable_surface   - magenta
        [139, 137, 137],  # other_flat          - gray
        [75,  0,   75 ],  # sidewalk            - dark purple
        [150, 240, 80 ],  # terrain             - light green
        [230, 230, 250],  # manmade             - lavender
        [0,   175, 0  ],  # vegetation          - green
    ], dtype=torch.float32)

    def __init__(self,
                 in_channels: int = 256,
                 mid_channels: int = 128,
                 num_classes: int = 17,
                 dropout: float=0.1
                 ):
        super().__init__()

        self.num_classes = num_classes

        self.mlp = nn.Sequential(

            # Layer 1
            nn.Linear(in_channels, mid_channels, bias= False),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            # Layer 2
            nn.Linear(mid_channels, num_classes),
        )
     

    def forward(self,
                features: torch.Tensor,
                inverse_indices: Optional[torch.Tensor] = None,
                num_points: Optional[int] = None,
                )->Optional[dict]:
        """
        Args:
            features        : (V, C)  PTv3 voxel features
            inverse_indices : (N,)    maps each original point to its voxel ID
                              produced by Voxelizer._group_and_encode
                              None during training if point labels not needed
            num_points      : N       number of original points
                              required if inverse_indices is provided

        Returns:
            dict with keys:
                "voxel_logits"  : (V, num_classes)  raw scores per voxel
                "voxel_labels"  : (V,)  argmax labels per voxel (inference)
                "point_labels"  : (N,)  labels propagated to points (inference)
                                  only present if inverse_indices is provided
                "point_colors"  : (N, 3) RGB colors per point for visualization
                                  only present if inverse_indices is provided
        """

        # MLP (V, num_classes)

        voxel_logits = self.mlp(features)

        # predicted class per voxel

        voxel_labels = voxel_logits.argmax(dim=1) # (V,)

        results = {
            "voxel_logits" : voxel_logits,
            "voxel_labels" : voxel_labels
        }
        # Propagate labels back to oriinal points

        if inverse_indices is not None and num_points is not None:
            point_labels = self._propagate_to_points(
                voxel_labels, inverse_indices, num_points
            )
            results["point_labels"] = point_labels

            # Assign colors in RGB
            colors = self.CLASS_COLORS.to(features.device)
            results["point_colors"] = colors[point_labels] # (N, 3)

        return results
    
    def _propagate_to_points(
            self,
            voxel_labels: torch.Tensor,
            inverse_indices: torch.Tensor,
            num_points: int
    ) -> Optional[torch.Tensor]:
        """
        Propagates voxel labels to original points.

        Each point inherits the label of its voxel.
        Points filtered out during voxelization (out-of-range) receive
        label 0 (noise) by default.

        Args:
            voxel_labels    : (V,)  one label per voxel
            inverse_indices : (N,)  voxel ID for each point (0..V-1)
                              points with ID >= V were filtered -- get label 0
            num_points      : N

        Returns:
            point_labels : (N,) one label per original point
        """
        device = voxel_labels.device

        # allocate points labels default 0 to noise for filtered points
        point_labels  = torch.zeros(num_points, dtype=torch.long, device = device)

        # propagate valid points

        valid_mask = inverse_indices < voxel_labels.shape[0]
        valid_idx = inverse_indices[valid_mask]

        point_labels[valid_mask] = voxel_labels[valid_idx]

        return point_labels #(N,)
    
    @staticmethod
    def build_loss(
        class_weights: Optional[torch.Tensor ]= None,
        ignore_index:  int          = 0,
    ) -> nn.CrossEntropyLoss:
        """
        Builds the segmentation loss function.

        Args:
            class_weights : (num_classes,) inverse-frequency weights
                            if None, uniform weighting
            ignore_index  : class index to ignore in loss (default 0 = noise)
                            noise points have unreliable annotations

        Returns:
            nn.CrossEntropyLoss configured for segmentation
        """
        return nn.CrossEntropyLoss(
            weight       = class_weights,
            ignore_index = ignore_index,
            reduction    = "mean",
        )

