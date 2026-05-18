"""
centerpoint_loss.py -- CenterPoint detection losses for AdPerception.

Implements the exact losses from the CenterPoint paper (Yin et al., 2021):
  - Heatmap : CornerNet focal loss with Gaussian target rendering
  - Offset   : L1 at GT center voxels only
  - Height   : L1 at GT center voxels only
  - Dims     : L1 on log(l,w,h) at GT center voxels only
  - Rotation : L1 on (sin, cos) at GT center voxels only
  - Velocity : L1 at GT center voxels only (zero target -- no GT vel in loader)

Loss weights from the paper:
  heatmap=1.0, offset=0.25, height=1.0, dims=1.0, rotation=1.0, velocity=0.25

References:
  CenterPoint (tianweiy/CenterPoint, det3d/models/bbox_heads/center_head.py)
  CornerNet   (gaussian_radius + focal loss formulation)
"""

import math
from typing import Tuple, Dict

import torch
import torch.nn.functional as F


# -----------------------------------------------------------------------------------------
# Gaussian heatmap helpers
# -----------------------------------------------------------------------------------------

def gaussian_radius(h_vox: float, w_vox: float, min_overlap: float = 0.1) -> int:
    """
    CornerNet Gaussian radius in voxel units.

    Computes the minimum radius such that a predicted center within `radius`
    voxels of the GT center achieves at least `min_overlap` IoU with the GT box.

    Args:
        h_vox       : GT box length (forward extent) in voxel units
        w_vox       : GT box width (lateral extent) in voxel units
        min_overlap : minimum IoU threshold (default 0.1, same as CenterPoint)

    Returns:
        radius : int >= 1
    """
    h, w = h_vox, w_vox

    a1 = 1.0
    b1 = h + w
    c1 = h * w * (1 - min_overlap) / (1 + min_overlap)
    sq1 = math.sqrt(max(0.0, b1 ** 2 - 4 * a1 * c1))
    r1 = (b1 - sq1) / 2.0

    a2 = 4.0
    b2 = 2.0 * (h + w)
    c2 = (1 - min_overlap) * h * w
    sq2 = math.sqrt(max(0.0, b2 ** 2 - 4 * a2 * c2))
    r2 = (b2 - sq2) / 4.0

    a3 = 4.0 * min_overlap
    b3 = -2.0 * min_overlap * (h + w)
    c3 = (min_overlap - 1.0) * h * w
    sq3 = math.sqrt(max(0.0, b3 ** 2 - 4 * a3 * c3))
    r3 = (b3 + sq3) / (2.0 * a3) if a3 > 0 else r1

    return max(1, int(min(r1, r2, r3)))


def _draw_gaussian(
    heatmap: torch.Tensor,   # (H, W) float32, modified in-place
    cy: int,
    cx: int,
    radius: int,
) -> None:
    """
    Draw a 2D Gaussian peak at (cy, cx) into `heatmap` in-place.

    Uses element-wise maximum so overlapping GT boxes keep the highest value
    at each cell (standard CornerNet / CenterPoint behaviour).

    sigma = radius / 3  →  value drops to e^{-0.5} ≈ 0.6 at 1σ,
                           to e^{-4.5} ≈ 0.01 at 3σ = radius.
    """
    H, W = heatmap.shape
    sigma = radius / 3.0

    y0 = max(0, cy - radius)
    y1 = min(H, cy + radius + 1)
    x0 = max(0, cx - radius)
    x1 = min(W, cx + radius + 1)

    if y0 >= y1 or x0 >= x1:
        return

    yy = torch.arange(y0, y1, device=heatmap.device, dtype=torch.float32) - cy
    xx = torch.arange(x0, x1, device=heatmap.device, dtype=torch.float32) - cx
    gy, gx = torch.meshgrid(yy, xx, indexing="ij")
    gaussian = torch.exp(-(gy ** 2 + gx ** 2) / (2.0 * sigma ** 2))

    heatmap[y0:y1, x0:x1] = torch.maximum(heatmap[y0:y1, x0:x1], gaussian)


def render_gt_heatmap(
    gt_boxes:   torch.Tensor,   # (M, 7)  [cx, cy, cz, l, w, h, yaw]
    gt_labels:  torch.Tensor,   # (M,)    int64 class indices
    num_classes: int,
    H: int,
    W: int,
    dx: float,
    dy: float,
    x_min: float,
    y_min: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Render Gaussian GT heatmap from GT boxes.

    Args:
        gt_boxes    : (M, 7)  [cx, cy, cz, l, w, h, yaw] in ego frame (meters)
        gt_labels   : (M,)    class index per GT box
        num_classes : K       number of detection classes
        H, W        : BEV grid height (Gy) and width (Gx)
        dx, dy      : voxel size in x, y (meters)
        x_min, y_min: BEV origin (meters)

    Returns:
        heatmap  : (K, H, W) float32 Gaussian target, values in [0, 1]
        pos_ys   : (M',) long  y-grid indices of valid GT centers
        pos_xs   : (M',) long  x-grid indices of valid GT centers
        targets  : dict of regression targets at valid GT centers
            "offset"   : (M', 2)  [dx_subvoxel, dy_subvoxel] in [0,1)
            "height"   : (M',)    cz in meters
            "log_dims" : (M', 3)  [log l, log w, log h]
            "rotation" : (M', 2)  [sin yaw, cos yaw]
            "velocity" : (M', 2)  zeros (no GT velocity in loader)
    """
    device = gt_boxes.device
    M      = gt_boxes.shape[0]

    heatmap = torch.zeros(num_classes, H, W, dtype=torch.float32, device=device)

    if M == 0:
        empty_long  = torch.zeros(0, dtype=torch.long,    device=device)
        empty_float = torch.zeros(0, dtype=torch.float32, device=device)
        return heatmap, empty_long, empty_long, {
            "offset"   : torch.zeros(0, 2, device=device),
            "height"   : empty_float,
            "log_dims" : torch.zeros(0, 3, device=device),
            "rotation" : torch.zeros(0, 2, device=device),
            "velocity" : torch.zeros(0, 2, device=device),
        }

    cx  = gt_boxes[:, 0]
    cy  = gt_boxes[:, 1]
    cz  = gt_boxes[:, 2]
    l   = gt_boxes[:, 3].clamp(min=0.1)
    w   = gt_boxes[:, 4].clamp(min=0.1)
    h   = gt_boxes[:, 5].clamp(min=0.1)
    yaw = gt_boxes[:, 6]

    # continuous grid positions
    ix_f = (cx - x_min) / dx   # float x-index (Gx axis)
    iy_f = (cy - y_min) / dy   # float y-index (Gy axis)

    # integer voxel centres (clamped for safety)
    ix = ix_f.long().clamp(0, W - 1)
    iy = iy_f.long().clamp(0, H - 1)

    # sub-voxel offset regression target (fractional part, in [0,1))
    offset_x = ix_f - ix.float()
    offset_y = iy_f - iy.float()

    # valid = box centre falls inside the BEV grid
    valid = (ix_f >= 0) & (ix_f < W) & (iy_f >= 0) & (iy_f < H)

    # Draw Gaussians for each valid GT box
    for m in range(M):
        if not valid[m]:
            continue
        cls = int(gt_labels[m].item())
        if cls < 0 or cls >= num_classes:
            continue

        ix_m = int(ix[m].item())
        iy_m = int(iy[m].item())

        # Gaussian radius from box footprint in voxel units
        r = gaussian_radius(
            h_vox = float(l[m].item()) / dx,
            w_vox = float(w[m].item()) / dy,
        )
        _draw_gaussian(heatmap[cls], iy_m, ix_m, r)

    # Regression targets at valid GT centers only
    pos_ys = iy[valid]
    pos_xs = ix[valid]

    targets = {
        "offset"   : torch.stack([offset_x[valid], offset_y[valid]], dim=1),   # (M', 2)
        "height"   : cz[valid],                                                  # (M',)
        "log_dims" : torch.stack([l[valid].log(), w[valid].log(), h[valid].log()], dim=1),  # (M', 3)
        "rotation" : torch.stack([yaw[valid].sin(), yaw[valid].cos()], dim=1),  # (M', 2)
        "velocity" : torch.zeros(int(valid.sum().item()), 2, device=device),    # (M', 2)
    }

    return heatmap, pos_ys, pos_xs, targets


# -----------------------------------------------------------------------------------------
# Loss functions
# -----------------------------------------------------------------------------------------

def focal_loss_cornernet(
    pred: torch.Tensor,   # (K, H, W)  sigmoid-activated heatmap predictions
    gt:   torch.Tensor,   # (K, H, W)  Gaussian target in [0, 1]
    alpha: float = 2.0,
    beta:  float = 4.0,
) -> torch.Tensor:
    """
    CornerNet focal loss for CenterPoint heatmap supervision.

    Positive positions (gt == 1): standard focal loss -(1-p)^α log(p)
    Negative positions           : down-weighted by (1-gt)^β to reduce
                                   penalty near GT centers where gt ∈ (0,1).

    Normalised by the number of GT objects (positive peaks), not by H*W,
    so the loss magnitude does not depend on BEV resolution.

    Args:
        pred  : (K, H, W) float32 -- sigmoid(heatmap logits)
        gt    : (K, H, W) float32 -- Gaussian target from render_gt_heatmap
        alpha : focal exponent for positive loss (default 2)
        beta  : penalty-reduction exponent for negatives (default 4)

    Returns:
        scalar loss tensor
    """
    pred = pred.clamp(1e-6, 1 - 1e-6)

    pos_mask = (gt == 1.0)
    num_pos  = pos_mask.sum().float().clamp(min=1.0)

    pos_loss = -((1 - pred) ** alpha) * torch.log(pred) * pos_mask.float()
    neg_loss = (
        -((1 - gt) ** beta)
        * (pred ** alpha)
        * torch.log(1 - pred)
        * (~pos_mask).float()
    )

    return (pos_loss.sum() + neg_loss.sum()) / num_pos


def centerpoint_loss(
    preds:       dict,
    gt_boxes:    torch.Tensor,   # (M, 7)  [cx, cy, cz, l, w, h, yaw]
    gt_labels:   torch.Tensor,   # (M,)
    voxel_size:  list,
    point_range: list,
    H: int,
    W: int,
) -> Tuple[torch.Tensor, dict]:
    """
    Full CenterPoint detection loss (single sample, batch index 0).

    Loss weights:
        heatmap=1.0, offset=0.25, height=1.0, dims=1.0, rotation=1.0, velocity=0.25

    Args:
        preds       : raw head outputs from CenterPointHead(decode=False)
                      keys: heatmap (1,K,H,W), offset (1,2,H,W), height (1,1,H,W),
                            dims (1,3,H,W), rotation (1,2,H,W), velocity (1,2,H,W)
        gt_boxes    : (M, 7) ego-frame GT boxes
        gt_labels   : (M,)   class indices
        voxel_size  : [dx, dy, dz]
        point_range : [x_min, y_min, z_min, x_max, y_max, z_max]
        H, W        : BEV grid height, width

    Returns:
        total_loss : scalar tensor
        loss_dict  : dict with per-head loss values (floats)
    """
    dx    = voxel_size[0]
    dy    = voxel_size[1]
    x_min = point_range[0]
    y_min = point_range[1]
    K     = preds["heatmap"].shape[1]

    # Render GT heatmap and regression targets
    gt_hm, pos_ys, pos_xs, targets = render_gt_heatmap(
        gt_boxes, gt_labels, K, H, W, dx, dy, x_min, y_min
    )

    # --- heatmap focal loss ------------------------------------------------------
    pred_hm = torch.sigmoid(preds["heatmap"][0])   # (K, H, W)
    loss_hm = focal_loss_cornernet(pred_hm, gt_hm)

    loss_dict = {"heatmap": loss_hm.item()}

    if pos_ys.shape[0] == 0:
        # No valid GT boxes in this frame -- only heatmap loss
        return loss_hm, loss_dict

    # --- regression losses at GT center positions ----------------------------
    ys, xs = pos_ys, pos_xs   # (M',)

    pred_off  = preds["offset"]  [0, :, ys, xs].T   # (M', 2)
    pred_hgt  = preds["height"]  [0, 0, ys, xs]     # (M',)
    pred_dims = preds["dims"]    [0, :, ys, xs].T    # (M', 3)
    pred_rot  = preds["rotation"][0, :, ys, xs].T   # (M', 2)
    pred_vel  = preds["velocity"][0, :, ys, xs].T   # (M', 2)

    loss_off  = F.l1_loss(pred_off,  targets["offset"])
    loss_hgt  = F.l1_loss(pred_hgt,  targets["height"])
    loss_dims = F.l1_loss(pred_dims, targets["log_dims"])
    loss_rot  = F.l1_loss(pred_rot,  targets["rotation"])
    loss_vel  = F.l1_loss(pred_vel,  targets["velocity"])

    total = (
        1.00 * loss_hm   +
        0.25 * loss_off  +
        1.00 * loss_hgt  +
        1.00 * loss_dims +
        1.00 * loss_rot  +
        0.25 * loss_vel
    )

    loss_dict.update({
        "offset"  : loss_off .item(),
        "height"  : loss_hgt .item(),
        "dims"    : loss_dims.item(),
        "rotation": loss_rot .item(),
        "velocity": loss_vel .item(),
    })

    return total, loss_dict
