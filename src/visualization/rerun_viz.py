"""
visualization.py -- Rerun-based visualization for AdPerception.

Logs LiDAR point clouds, semantic segmentation colors, and tracked
dynamic objects (vehicles, pedestrians, cyclists) to Rerun.

Rerun viewer opens automatically on first use.
Timeline navigation allows scrubbing through frames after recording.

Usage:
    viz = Visualizer()
    viz.update(points, out, tracks, frame_idx=i)
    viz.close()
"""

from typing import List

import numpy as np
import torch
import rerun as rr
import rerun.blueprint as rrb

from src.tracking.simpletrack_wrapper import Track

# maximum number of points sent to Rerun per frame
# keeps the viewer fluid
MAX_DISPLAY_PTS = 10000

# detection class names -- index = CenterPoint class id
DET_CLASSES = [
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
]

# detection class colors -- RGB uint8
DET_COLORS = np.array([
    [0,   150, 245],  # car          -- blue
    [160, 32,  240],  # truck        -- purple
    [135, 60,  0  ],  # construction -- brown
    [255, 255, 0  ],  # bus          -- yellow
    [255, 192, 203],  # trailer      -- pink
    [255, 120, 50 ],  # barrier      -- orange
    [200, 180, 0  ],  # motorcycle   -- gold
    [255, 0,   0  ],  # bicycle      -- red
    [255, 0,   255],  # pedestrian   -- magenta
    [255, 240, 150],  # traffic_cone -- cream
], dtype=np.uint8)


class Visualizer:
    """
    Rerun-based real-time visualizer for AdPerception.

    Opens the Rerun viewer automatically and logs each frame
    to the timeline for real-time display and post-hoc scrubbing.

    Args:
        app_name  : Rerun application name shown in the viewer
        ego_cx    : longitudinal center of the ego vehicle in ego frame (m).
                    In nuScenes the ego frame origin is at the rear axle;
                    the LiDAR x-offset (≈0.944 m) is a good proxy for the
                    vehicle's longitudinal centre.  Defaults to 0.944.
    """

    def __init__(self, app_name: str = "AdPerception", ego_cx: float = 0.944):

        self._ego_cx = ego_cx   # longitudinal centre of ego vehicle in ego frame

        rr.init(app_name, spawn=True)

        # layout: 3D perspective view (left) + BEV top-down 2D (right)
        blueprint = rrb.Horizontal(
            rrb.Spatial3DView(
                name   = "AdPerception 3D view",
                origin = "/world",
            ),
            rrb.Spatial2DView(
                name   = "AdPerception BEV (top-down)",
                origin = "/bev",
            ),
        )
        rr.send_blueprint(blueprint)

        # nuScenes LiDAR convention: X=forward, Y=left, Z=up (FLU)
        rr.log("world", rr.ViewCoordinates.FLU, static=True)

        # world origin axes (X red, Y green, Z blue)
        rr.log(
            "world/axes",
            rr.Arrows3D(
                origins = [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                vectors = [[5, 0, 0], [0, 5, 0], [0, 0, 5]],
                colors  = [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            ),
            static=True,
        )

        # BEV range rings at 20 m / 40 m / 60 m  (reference circle)
        angles = np.linspace(0, 2 * np.pi, 128)
        for radius, alpha in [(20, 100), (40, 70), (60, 50)]:
            ring = np.column_stack([
                radius * np.cos(angles),
                radius * np.sin(angles),
            ]).astype(np.float32)
            rr.log(
                f"bev/range/{radius}m",
                rr.LineStrips2D(
                    [np.vstack([ring, ring[:1]])],   # close the circle
                    colors = [[alpha, alpha, alpha]],
                    radii  = 0.15,
                ),
                static=True,
            )

        # BEV forward direction tick (+X = forward)
        rr.log(
            "bev/forward",
            rr.Arrows2D(
                origins = [[0.0, 0.0]],
                vectors = [[6.0, 0.0]],
                colors  = [[0, 255, 136]],
                radii   = 0.2,
            ),
            static=True,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    # index of driveable_surface in the 17-class nuScenes seg mapping
    _DRIVEABLE_CLASS = 11

    def update(
        self,
        points:    torch.Tensor,
        out:       dict,
        tracks:    List[Track],
        frame_idx: int = 0,
        gt_boxes:  "torch.Tensor | None" = None,
        gt_labels: "torch.Tensor | None" = None,
    ) -> None:
        """
        Logs one frame to Rerun in the ego vehicle frame.

        Args:
            points    : (N, 4)  LiDAR points [x, y, z, intensity] in ego frame
            out       : pipeline output dict -- uses "point_colors" for semantics
            tracks    : active Track objects from SimpleTrackWrapper
            frame_idx : current frame index -- sets the timeline position
            gt_boxes  : (M, 7) optional GT boxes [cx,cy,cz,l,w,h,yaw] for overlay
            gt_labels : (M,)   optional GT class indices for overlay
        """
        rr.set_time_sequence("frame", frame_idx)

        pts_np    = points.cpu().numpy()
        pt_colors = out["point_colors"].cpu().numpy().astype(np.uint8)

        # uniform stride subsample for display
        stride      = max(1, len(pts_np) // MAX_DISPLAY_PTS)
        pts_display = pts_np[::stride]
        col_display = pt_colors[::stride]

        self._log_ego()
        self._log_points(pts_display, col_display)
        self._log_tracks(tracks)

        # navigable zone: highlight driveable_surface points on BEV
        if "point_labels" in out:
            self._log_driveable(pts_np, out["point_labels"].cpu().numpy())
        else:
            rr.log("bev/driveable", rr.Clear(recursive=False))

        # GT boxes overlay (white) for prediction vs GT comparison
        if gt_boxes is not None and gt_labels is not None:
            self._log_gt_boxes(gt_boxes.cpu(), gt_labels.cpu())
        else:
            rr.log("world/gt_boxes", rr.Clear(recursive=True))
            rr.log("bev/gt_boxes",   rr.Clear(recursive=True))

    def _log_gt_boxes(
        self,
        gt_boxes:  "torch.Tensor",   # (M, 7) [cx,cy,cz,l,w,h,yaw] CPU
        gt_labels: "torch.Tensor",   # (M,)   CPU
    ) -> None:
        """
        Logs nuScenes GT boxes to Rerun for comparison with predictions.

        Colour: white (to distinguish from coloured predicted tracks).
        Paths : world/gt_boxes (3D)  bev/gt_boxes (2D BEV outline)
        """
        M = gt_boxes.shape[0]
        if M == 0:
            rr.log("world/gt_boxes", rr.Clear(recursive=True))
            rr.log("bev/gt_boxes",   rr.Clear(recursive=True))
            return

        gt_np  = gt_boxes.numpy()
        lbl_np = gt_labels.numpy()

        # ── 3D boxes ──────────────────────────────────────────────────────────
        centers    = gt_np[:, :3]
        half_sizes = gt_np[:, 3:6] / 2.0
        yaws       = gt_np[:, 6]
        half_yaws  = yaws / 2.0
        quats_xyzw = np.stack([
            np.zeros_like(yaws),
            np.zeros_like(yaws),
            np.sin(half_yaws),
            np.cos(half_yaws),
        ], axis=1)
        labels_3d = [
            f"GT:{DET_CLASSES[int(lbl_np[m]) % len(DET_CLASSES)]}"
            for m in range(M)
        ]

        rr.log(
            "world/gt_boxes",
            rr.Boxes3D(
                centers    = centers,
                half_sizes = half_sizes,
                rotations  = [rr.Quaternion(xyzw=q) for q in quats_xyzw],
                colors     = [[255, 255, 255]] * M,   # white
                labels     = labels_3d,
            ),
        )

        # ── BEV outlines (rotated, Y-flipped) ────────────────────────────────
        bev_strips = []
        for m in range(M):
            cx, cy = float(gt_np[m, 0]), float(gt_np[m, 1])
            l,  w  = float(gt_np[m, 3]), float(gt_np[m, 4])
            yaw    = float(gt_np[m, 6])
            bev_strips.append(self._bev_box_corners(cx, cy, l, w, yaw))

        rr.log(
            "bev/gt_boxes",
            rr.LineStrips2D(
                bev_strips,
                colors = [[255, 255, 255]] * M,   # white
                radii  = 0.06,
                labels = labels_3d,
            ),
        )

    def _log_driveable(
        self,
        pts_np:       np.ndarray,   # (N, 4) all LiDAR points
        point_labels: np.ndarray,   # (N,)   per-point seg class
    ) -> None:
        """
        Highlights driveable_surface points (class 11) in the BEV.

        Colour: bright green (#80FF80) to mark the navigable zone.
        Logged to bev/driveable -- separate layer from the coloured point cloud.
        """
        drv_mask = (point_labels == self._DRIVEABLE_CLASS)
        if not drv_mask.any():
            rr.log("bev/driveable", rr.Clear(recursive=False))
            return

        drv_pts = pts_np[drv_mask]
        bev_pos = np.column_stack([drv_pts[:, 0], -drv_pts[:, 1]])

        # subsample to keep viewer fluid
        stride = max(1, len(bev_pos) // MAX_DISPLAY_PTS)
        bev_pos = bev_pos[::stride]

        rr.log(
            "bev/driveable",
            rr.Points2D(
                bev_pos,
                colors = [128, 255, 128],   # bright green
                radii  = 0.06,
            ),
        )

    def close(self) -> None:
        """No-op -- Rerun viewer stays open for timeline scrubbing."""
        pass

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bev_box_corners(
        cx: float, cy: float, l: float, w: float, yaw: float
    ) -> np.ndarray:
        """
        Returns the 5 BEV corners (closed strip) of an oriented box.

        Args:
            cx, cy : center in ego frame (meters)
            l      : length  -- extent along vehicle forward axis
            w      : width   -- extent along vehicle lateral axis
            yaw    : heading in radians (nuScenes convention: CCW from +X)

        Returns:
            (5, 2) float32 array in BEV canvas coords (U=ego_x, V=-ego_y)
        """
        hl, hw = l / 2.0, w / 2.0
        # four corners in vehicle-local frame (forward=+X, left=+Y)
        local = np.array([
            [+hl, +hw],
            [+hl, -hw],
            [-hl, -hw],
            [-hl, +hw],
        ], dtype=np.float32)
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s], [s, c]], dtype=np.float32)
        corners_ego = local @ R.T + np.array([cx, cy], dtype=np.float32)
        # BEV canvas: U = ego_x, V = -ego_y
        bev = np.column_stack([corners_ego[:, 0], -corners_ego[:, 1]])
        return np.vstack([bev, bev[:1]])  # close the strip

    def _log_ego(self) -> None:
        """Logs the ego vehicle box (Renault Zoe, nuScenes convention).

        nuScenes ego frame: origin at rear-axle center, ground level.
        The LiDAR x-offset (≈ 0.944 m) is used as the longitudinal center
        of the vehicle body (rear bumper ≈ -1.06 m, front ≈ +2.94 m).
        Dimensions: 4.08 m × 1.77 m × 1.56 m  → half (2.04, 0.89, 0.78).
        """
        cx = self._ego_cx
        rr.log(
            "world/ego",
            rr.Boxes3D(
                centers    = [[cx, 0.0, 0.78]],   # mid-height of Renault Zoe
                half_sizes = [[2.04, 0.89, 0.78]],
                colors     = [[0, 255, 136]],
                labels     = ["ego"],
            ),
        )
        ego_corners = self._bev_box_corners(cx, 0.0, 4.08, 1.77, 0.0)
        rr.log(
            "bev/ego",
            rr.LineStrips2D(
                [ego_corners],
                colors = [[0, 255, 136]],
                radii  = 0.12,
            ),
        )

    def _log_points(
        self,
        pts_np:    np.ndarray,
        pt_colors: np.ndarray,
    ) -> None:
        """
        Logs LiDAR point cloud to 3D view and BEV.

        3D: Points3D under world/lidar  (XYZ + semantic color)
        BEV: Points2D under bev/lidar   (XY projection, Y-flipped)
        """
        rr.log(
            "world/lidar",
            rr.Points3D(
                pts_np[:, :3],
                colors = pt_colors,
                radii  = 0.05,
            ),
        )
        # BEV: +U = ego X (forward), +V = -ego Y (right)
        bev_pos = np.column_stack([pts_np[:, 0], -pts_np[:, 1]])
        rr.log(
            "bev/lidar",
            rr.Points2D(
                bev_pos,
                colors = pt_colors,
                radii  = 0.05,
            ),
        )

    def _log_tracks(self, tracks: List[Track]) -> None:
        """
        Logs tracked dynamic objects: 3D boxes, 2D BEV boxes, and BEV trails.

        Each track keeps its stable ID across frames.
        Trail = sequence of past (cx, cy) positions drawn as a line strip in BEV.
        """
        if not tracks:
            rr.log("world/tracks",  rr.Clear(recursive=True))
            rr.log("bev/tracks",    rr.Clear(recursive=True))
            rr.log("bev/trails",    rr.Clear(recursive=False))
            rr.log("bev/velocity",  rr.Clear(recursive=False))
            return

        # --- 3D boxes ---
        centers_3d = np.array([t.box[:3] for t in tracks], dtype=np.float32)
        half_sizes  = np.array([t.box[3:6] for t in tracks], dtype=np.float32) / 2.0
        yaws        = np.array([t.box[6] for t in tracks], dtype=np.float32)
        colors      = np.array([
            DET_COLORS[t.label % len(DET_COLORS)] for t in tracks
        ], dtype=np.uint8)
        labels_3d   = [
            f"{DET_CLASSES[t.label % len(DET_CLASSES)]}#{t.track_id}"
            for t in tracks
        ]

        half_yaws  = yaws / 2.0
        quats_xyzw = np.stack([
            np.zeros_like(yaws),
            np.zeros_like(yaws),
            np.sin(half_yaws),
            np.cos(half_yaws),
        ], axis=1)

        rr.log(
            "world/tracks",
            rr.Boxes3D(
                centers    = centers_3d,
                half_sizes = half_sizes,
                rotations  = [rr.Quaternion(xyzw=q) for q in quats_xyzw],
                colors     = colors,
                labels     = labels_3d,
            ),
        )

        # --- BEV boxes (rotated, Y-flipped) ---
        # Boxes2D is axis-aligned -- use LineStrips2D with actual yaw so that
        # a car parked at 45° is drawn correctly and does not falsely overlap
        # with the ego box or adjacent vehicles.
        bev_strips = []
        for t in tracks:
            cx, cy = float(t.box[0]), float(t.box[1])
            l,  w  = float(t.box[3]), float(t.box[4])
            yaw    = float(t.box[6])
            bev_strips.append(self._bev_box_corners(cx, cy, l, w, yaw))

        rr.log(
            "bev/tracks",
            rr.LineStrips2D(
                bev_strips,
                colors = colors,
                radii  = 0.1,
                labels = labels_3d,
            ),
        )

        # --- BEV velocity arrows ---
        vel_origins = []
        vel_vectors = []
        vel_colors  = []
        for t in tracks:
            vx, vy = float(t.velocity[0]), float(t.velocity[1])
            speed = (vx**2 + vy**2) ** 0.5
            if speed < 0.3:   # skip near-static objects
                continue
            cx, cy = float(t.box[0]), float(t.box[1])
            vel_origins.append([cx, -cy])
            vel_vectors.append([vx * 2.0, -vy * 2.0])   # scale for visibility
            vel_colors.append(DET_COLORS[t.label % len(DET_COLORS)].tolist())

        if vel_origins:
            rr.log(
                "bev/velocity",
                rr.Arrows2D(
                    origins = np.array(vel_origins, dtype=np.float32),
                    vectors = np.array(vel_vectors, dtype=np.float32),
                    colors  = vel_colors,
                    radii   = 0.08,
                ),
            )
        else:
            rr.log("bev/velocity", rr.Clear(recursive=False))

        # --- BEV trails (all tracks in a single LineStrips2D call) ---
        trail_strips = []
        trail_colors = []
        for t in tracks:
            if len(t.trail) < 2:
                continue
            trail_arr = np.array(t.trail, dtype=np.float32)
            trail_bev = np.column_stack([trail_arr[:, 0], -trail_arr[:, 1]])
            trail_strips.append(trail_bev)
            trail_colors.append(DET_COLORS[t.label % len(DET_COLORS)].tolist())

        if trail_strips:
            rr.log(
                "bev/trails",
                rr.LineStrips2D(
                    trail_strips,
                    colors = trail_colors,
                    radii  = 0.04,
                ),
            )
        else:
            rr.log("bev/trails", rr.Clear(recursive=False))
