"""
simpletrack_wrapper.py -- SimpleTrack MOT wrapper for AdPerception.

Wraps mot_3d.MOTModel behind a clean interface that consumes
CenterPoint outputs and produces Track objects for visualization.

SimpleTrack pipeline per frame:
    1. Kalman prediction  : each track predicts position at t
    2. GIoU matching      : Hungarian on cost = 1 - GIoU
    3. Track update       : matched tracks updated with detection
    4. Track birth        : unmatched detections -> new tracks
    5. Track death        : unmatched tracks for max_age frames -> removed

BBox format conversion:
    CenterPoint : [cx, cy, cz, l, w, h, sin_t, cos_t]
    SimpleTrack : [x,  y,  z,  o, l, w, h    ]
                                 ^
                                 yaw = arctan2(sin_t, cos_t)

FrameData expects a list of raw numpy arrays -- it calls BBox.array2bbox
internally. Do NOT pass BBox objects directly or it will crash.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import os

import numpy as np
import torch
import yaml

from mot_3d.data_protos.bbox import BBox
from mot_3d.mot import MOTModel
from mot_3d.frame_data import FrameData


@dataclass
class Track:
    """
    Single track output -- consumed by visualization and cooperative modules.

    Attributes:
        track_id : unique integer ID stable across frames
        box      : (7,) [cx, cy, cz, l, w, h, yaw] in ego frame
        score    : detection confidence score
        label    : class index (0=car ... 9=traffic_cone)
        velocity : (2,) [vx, vy] in m/s
        trail    : list of past (cx, cy) positions for BEV trail rendering
        age      : number of frames this track has been active
    """
    track_id : int
    box      : np.ndarray
    score    : float
    label    : int
    velocity : np.ndarray
    trail    : List = field(default_factory=list)
    age      : int  = 1


class SimpleTrackWrapper:
    """
    Stateful MOT wrapper around SimpleTrack (mot_3d).

    One MOTModel per detection class -- prevents cross-class associations
    (a car cannot be matched to a pedestrian even if their centers are close).

    Must call reset() between scenes to avoid ghost tracks.

    Args:
        num_classes   : number of detection classes (nuScenes = 10)
        max_trail_len : max past positions stored per track for BEV rendering
        config_path   : path to SimpleTrack YAML config
                        defaults to nu_configs/giou.yaml
    """

    DET_CLASSES = [
        "car", "truck", "construction_vehicle", "bus", "trailer",
        "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
    ]

    def __init__(
        self,
        num_classes:   int           = 10,
        max_trail_len: int           = 20,
        config_path:   Optional[str] = None,
    ):
        self.num_classes   = num_classes
        self.max_trail_len = max_trail_len
        self._timestamp: float = 0.0

        # load nuScenes GIoU config from SimpleTrack
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__),
                "../../third_party/SimpleTrack/configs/nu_configs/giou.yaml",
            )
        with open(os.path.abspath(config_path), "r") as f:
            self.config = yaml.safe_load(f)

        # trail and age history -- keyed by (cls_idx, track_id)
        self._trails: dict = {}
        self._ages:   dict = {}

        # one MOTModel per class
        self._init_models()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        boxes:    torch.Tensor,
        scores:   torch.Tensor,
        labels:   torch.Tensor,
        velocity: torch.Tensor,
    ) -> List[Track]:
        """
        Runs one frame of tracking across all detection classes.

        Args:
            boxes    : (M, 8) [cx, cy, cz, l, w, h, sin_t, cos_t]
            scores   : (M,)   confidence scores
            labels   : (M,)   class indices
            velocity : (M, 2) [vx, vy] from CenterPoint velocity head

        Returns:
            list of active Track objects after association
        """
        boxes_np    = boxes.cpu().numpy().astype(np.float64)
        scores_np   = scores.cpu().numpy().astype(np.float64)
        labels_np   = labels.cpu().numpy().astype(np.int32)
        velocity_np = velocity.cpu().numpy().astype(np.float64)

        # convert (sin_t, cos_t) -> yaw
        # CenterPoint encodes rotation as sin/cos for training stability
        # SimpleTrack expects yaw directly
        sin_t = boxes_np[:, 6]
        cos_t = boxes_np[:, 7]
        yaw   = np.arctan2(sin_t, cos_t)  # (M,)

        # reorder to SimpleTrack BBox format: [x, y, z, o, l, w, h]
        # CenterPoint output                : [cx, cy, cz, l, w, h, sin, cos]
        dets_st = np.stack([
            boxes_np[:, 0],  # x  = cx
            boxes_np[:, 1],  # y  = cy
            boxes_np[:, 2],  # z  = cz
            yaw,             # o  = yaw
            boxes_np[:, 3],  # l
            boxes_np[:, 4],  # w
            boxes_np[:, 5],  # h
        ], axis=1)  # (M, 7)

        active_tracks = []

        self._timestamp +=0.1

        for cls_idx in range(self.num_classes):

            cls_mask = labels_np == cls_idx

            if cls_mask.sum() == 0:
                # no detections for this class -- feed empty frame so the
                # tracker ages and eventually removes lost tracks
                dets_cls = []
            else:
                # append score as 8th column -- FrameData calls
                # BBox.array2bbox(row) internally which reads row[:7]
                # for geometry and row[7] for score
                # DO NOT pass BBox objects -- FrameData expects raw arrays
                scores_col = scores_np[cls_mask].reshape(-1, 1)
                dets_array = np.concatenate(
                    [dets_st[cls_mask], scores_col], axis=1
                )  # (K, 8)
                dets_cls = dets_array.tolist()  # list of lists -- FrameData converts
            
            # prepare velocities for aux_info

            if cls_mask.sum() == 0:
                velos = np.zeros((0,2), dtype=np.float64)
            else:
                velos = velocity_np[cls_mask]       #(K, 2)

            # FrameData wraps detections + ego pose for SimpleTrack
            # ego = identity because we work in ego vehicle frame
            frame_data = FrameData(
                dets       = dets_cls,
                ego        = np.eye(4),
                time_stamp = self._timestamp,
                det_types  = [0] * len(dets_cls),
                aux_info   = {"is_key_frame": True,
                              "velos" : velos,},
            )

            # run one frame of SimpleTrack
            # returns list of (BBox, track_id) tuples
            results = self.models[cls_idx].frame_mot(frame_data)

            for state, track_id, state_string, det_type in results:
                track = self._make_track(
                    state, int(track_id), cls_idx, velocity_np, labels_np, cls_mask
                )
                active_tracks.append(track)

        return active_tracks

    def reset(self) -> None:
        """
        Resets all trackers and clears trail history.
        Must be called between scenes to avoid ghost tracks from
        the previous scene polluting the new one.
        """
        self._init_models()
        self._trails.clear()
        self._ages.clear()
        self._timestamp = 0.0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_models(self) -> None:
        """Initializes one MOTModel per detection class."""
        self.models = [
            MOTModel(self.config) for _ in range(self.num_classes)
        ]

    def _make_track(
        self,
        bbox:       BBox,
        track_id:   int,
        cls_idx:    int,
        velocity_np: np.ndarray,
        labels_np:  np.ndarray,
        cls_mask:   np.ndarray,
    ) -> Track:
        """
        Converts a SimpleTrack (BBox, id) pair to our Track dataclass.

        SimpleTrack BBox convention : [x, y, z, o, l, w, h]
        Our Track box convention    : [cx, cy, cz, l, w, h, yaw]

        Velocity is taken from CenterPoint output for this class.
        If multiple detections exist for this class we use the mean velocity
        as an approximation -- in production the matched detection velocity
        would be used via the association indices.
        """
        # reorder BBox to our convention
        box = np.array([
            bbox.x, bbox.y, bbox.z,  # cx, cy, cz
            bbox.l, bbox.w, bbox.h,  # l, w, h
            bbox.o,                  # yaw
        ], dtype=np.float32)         # type: ignore

        score = float(bbox.s) if bbox.s is not None else 0.0 #type: ignore

        # mean velocity for this class as approximation
        if cls_mask.sum() > 0:
            vel = velocity_np[cls_mask].mean(axis=0).astype(np.float32)
        else:
            vel = np.zeros(2, dtype=np.float32)

        # update trail and age
        key = (cls_idx, track_id)
        # assert guarantees Pylance that fields are not None at this point
        assert bbox.x is not None and bbox.y is not None, "SimpleTrack returned empty BBox"

        cx, cy = float(bbox.x), float(bbox.y)

        if key not in self._trails:
            self._trails[key] = []
            self._ages[key]   = 0

        self._trails[key].append((cx, cy))
        if len(self._trails[key]) > self.max_trail_len:
            self._trails[key].pop(0)
        self._ages[key] += 1

        return Track(
            track_id = track_id,
            box      = box,
            score    = score,
            label    = cls_idx,
            velocity = vel,
            trail    = list(self._trails[key]),
            age      = self._ages[key],
        )