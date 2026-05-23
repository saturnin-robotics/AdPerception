"""
ptv3_wrapper.py - PTv3 backbone wrapper for AdPerception.

Bridges the gap between spconv SparseConvTensor (voxelizer output)
and the Pointcept Point dict format expected by PTv3.

Interface:
    input  : spconv.SparseConvTensor  (V, C_in)
    output : torch.Tensor             (V, C_out)  enriched voxel features
"""

from typing import Optional

import torch
import torch.nn as nn
import spconv.pytorch as spconv

from pointcept.models.point_transformer_v3 import PointTransformerV3
from pointcept.models.utils.structure import Point


class PTv3Wrapper(nn.Module):
    """
    Wraps PTv3 from Pointcept for use in AdPerception.

    Args:
        in_channels  : number of input feature channels (4 for x,y,z,intensity)
        out_channels : number of output feature channels consumed by heads
        ptv3_cfg     : dict of PTv3 constructor kwargs. If None, uses default.
    """

    def __init__(
        self,
        in_channels:  int            = 4,
        out_channels: int            = 256,
        ptv3_cfg:     Optional[dict] = None,
    ):
        super().__init__()

        default_cfg: dict = dict(
            in_channels        = in_channels,
            order              = ["z", "z-trans", "hilbert", "hilbert-trans"],
            stride             = [2, 2, 2, 2],
            enc_depths         = [3, 3, 3, 12, 3],
            enc_channels       = [48, 96, 192, 384, 512],
            enc_num_head       = [3, 6, 12, 24, 32],
            enc_patch_size     = [512, 512, 512, 512, 512],
            dec_depths         = [2, 2, 2, 2],
            dec_channels       = [64, 128, 192, 384],
            dec_num_head       = [4, 8, 12, 24],
            dec_patch_size     = [512, 512, 512, 512],
            mlp_ratio          = 4,
            qkv_bias           = True,
            qk_scale           = None,
            attn_drop          = 0.0,
            proj_drop          = 0.0,
            drop_path          = 0.3,
            shuffle_orders     = True,
            pre_norm           = True,
            enable_rpe         = False,
            enable_flash       = False,
            upcast_attention   = False,
            upcast_softmax     = False,
        )

        cfg: dict = ptv3_cfg if ptv3_cfg is not None else default_cfg

        self.backbone: nn.Module = PointTransformerV3(**cfg)  # type: ignore

        dec_out: int = int(cfg["dec_channels"][0])
        self.proj = nn.Linear(dec_out, out_channels)

    # =========================================================================
    # Public API
    # =========================================================================

    def forward(self, sparse_tensor: spconv.SparseConvTensor) -> torch.Tensor:
        """
        Args:
            sparse_tensor : spconv.SparseConvTensor produced by Voxelizer
                            features (V, C_in), indices (V, 4) [batch,Z,Y,X]

        Returns:
            torch.Tensor (V, C_out) - enriched voxel features ready for heads
        """
        point = self._to_point(sparse_tensor)

        # Force fp32 for all spconv operations inside PTv3.
        # spconv-cu121 backward crashes with mixed precision on H100
        # due to empty indices in Ampere fp16 kernels.
        with torch.cuda.amp.autocast(enabled=False):
            point.feat = point.feat.float()
            point = self.backbone(point)

        features = self.proj(point["feat"])
        return features

    # =========================================================================
    # Private helpers
    # =========================================================================

    def _to_point(self, sparse_tensor: spconv.SparseConvTensor) -> Point:
        """
        Convert a spconv SparseConvTensor to a Pointcept Point dict.

        spconv indices : (V, 4)  [batch, Z, Y, X]
        Pointcept coord: (V, 3)  [Z, Y, X] -- batch separated
        """
        indices = sparse_tensor.indices  # (V, 4) [batch, Z, Y, X]

        point = Point(
            feat      = sparse_tensor.features,
            coord     = indices[:, 1:].float(),
            batch     = indices[:, 0].long(),
            grid_size = torch.tensor(
                sparse_tensor.spatial_shape,
                dtype  = torch.float32,
                device = sparse_tensor.features.device,
            ),
        )

        return point
