"""
nuscenes_loader.py -- nuScenes LiDAR dataloader for AdPerception.

Loads LiDAR point clouds and 3D annotations from nuScenes train val.
Returns dicts consumed directly by LiDARPerceptionPipeline.

nuScenes coordinate conventions:
    - LiDAR points are stored in the LiDAR sensor frame
    - Annotations are stored in the global frame
    - We transform everything to the ego vehicle frame at sample time

File format:
    .pcd.bin -- binary float32, 5 values per point: x, y, z, intensity, ring
    We keep only (x, y, z, intensity) -- ring_index not used by PTv3
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes


class NuScenesLiDARDataset(Dataset):
    """
    PyTorch Dataset for nuScenes LiDAR perception.

    Returns one sample per nuScenes keyframe. Each sample contains
    the LiDAR point cloud in ego frame and GT 3D boxes for evaluation.

    Args:
        dataroot : path to nuScenes root (contains maps/, samples/, v1.0-*/)
        version  : dataset version ('v1.0-mini', 'v1.0-trainval', 'v1.0-test')
        split    : 'train' or 'val' -- determines which scenes to use
        verbose  : print nuScenes loading info
    """

    # nuScenes detection classes used by CenterPoint (10 classes)
    # index = class id fed to CenterPointHead
    DET_CLASSES = [
        "car", "truck", "construction_vehicle", "bus", "trailer",
        "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
    ]

    def __init__(
        self,
        dataroot: str,
        version:  str  = "v1.0-mini",
        split:    str  = "train",
        verbose:  bool = False,
    ):
        self.dataroot = dataroot
        self.version  = version
        self.split    = split

        # load nuScenes tables
        self.nusc = NuScenes(
            version  = version,
            dataroot = dataroot,
            verbose  = verbose,
        )

        # build class name -> index mapping
        self.class_to_idx = {c: i for i, c in enumerate(self.DET_CLASSES)}

        # collect all sample tokens for this split
        self.sample_tokens = self._get_split_tokens()

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.sample_tokens)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            dict with keys:
                "points"      : (N, 4) float32  [x, y, z, intensity] in ego frame
                "gt_boxes"    : (M, 7) float32  [cx, cy, cz, l, w, h, yaw] in ego frame
                "gt_labels"   : (M,)   int64    class index per GT box
                "sample_token": str    nuScenes sample token for evaluation
        """
        token  = self.sample_tokens[idx]
        sample = self.nusc.get("sample", token)

        # load LiDAR point cloud in ego frame
        points = self._load_lidar(sample)

        # load GT annotations in ego frame
        gt_boxes, gt_labels = self._load_annotations(sample)

        # ego-to-world transform for visualization
        ego_t, ego_r = self._get_ego_pose(sample)

        return {
            "points"          : torch.from_numpy(points).float(),
            "gt_boxes"        : torch.from_numpy(gt_boxes).float(),
            "gt_labels"       : torch.from_numpy(gt_labels).long(),
            "sample_token"    : token,
            "ego_translation" : torch.from_numpy(ego_t),  # (3,)
            "ego_rotation"    : torch.from_numpy(ego_r),  # (3, 3)
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_split_tokens(self) -> list:
        """
        Returns sample tokens for the requested split.

        For v1.0-mini all 404 samples are used for both train and val
        since mini has no official split file.
        For v1.0-trainval the official splits are used.
        """
        if self.version == "v1.0-mini":
            # mini has no split -- use all samples for both train and val
            return [s["token"] for s in self.nusc.sample]

        # official splits for trainval
        from nuscenes.utils.splits import create_splits_scenes
        splits     = create_splits_scenes()
        split_key  = "train" if self.split == "train" else "val"
        split_scenes = set(splits[split_key])

        tokens = []
        for sample in self.nusc.sample:
            scene = self.nusc.get("scene", sample["scene_token"])
            if scene["name"] in split_scenes:
                tokens.append(sample["token"])
        return tokens

    def _get_ego_pose(self, sample: dict):
        """
        Returns the ego-to-world transform for this sample.

        Returns:
            ego_t : (3,)   float32  translation [x, y, z] in world frame
            ego_r : (3, 3) float32  rotation matrix (ego -> world)
        """
        lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_data  = self.nusc.get("sample_data", lidar_token)
        ego_pose    = self.nusc.get("ego_pose", lidar_data["ego_pose_token"])
        ego_t       = np.array(ego_pose["translation"], dtype=np.float32)
        ego_r       = Quaternion(ego_pose["rotation"]).rotation_matrix.astype(np.float32)
        return ego_t, ego_r

    def _load_lidar(self, sample: dict) -> np.ndarray:
        """
        Loads LiDAR point cloud and transforms to ego vehicle frame.

        nuScenes stores points in the LiDAR sensor frame. We apply
        two transforms to reach the ego frame:
            1. lidar_to_ego : sensor mounting transform (fixed per scene)
            2. ego_to_global : not needed here -- we stay in ego frame

        Args:
            sample : nuScenes sample dict

        Returns:
            points : (N, 4) float32  [x, y, z, intensity] in ego frame
        """
        # get LiDAR sample data
        lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_data  = self.nusc.get("sample_data", lidar_token)

        # load binary point cloud -- 5 floats per point
        pcd_path = os.path.join(self.dataroot, lidar_data["filename"])
        scan     = np.fromfile(pcd_path, dtype=np.float32)
        points   = scan.reshape(-1, 5)[:, :4]  # keep x, y, z, intensity only

        # get calibration -- LiDAR sensor pose in ego frame
        calib = self.nusc.get(
            "calibrated_sensor", lidar_data["calibrated_sensor_token"]
        )

        # rotation and translation: LiDAR frame -> ego frame
        rot   = Quaternion(calib["rotation"]).rotation_matrix     # (3, 3)
        trans = np.array(calib["translation"])                     # (3,)

        # apply transform to xyz coordinates
        xyz_lidar = points[:, :3]                    # (N, 3)
        xyz_ego   = xyz_lidar @ rot.T + trans        # (N, 3)

        # reassemble with intensity
        points_ego = np.concatenate(
            [xyz_ego, points[:, 3:4]], axis=1       # (N, 4)
        )

        return points_ego.astype(np.float32)

    def _load_annotations(self, sample: dict):
        """
        Loads 3D GT boxes and transforms to ego vehicle frame.

        nuScenes annotations are stored in the global frame.
        We apply the inverse ego_to_global transform to reach ego frame.

        Args:
            sample : nuScenes sample dict

        Returns:
            gt_boxes  : (M, 7) float32  [cx, cy, cz, l, w, h, yaw]
            gt_labels : (M,)   int64    class index (-1 = ignore)
        """
        # get ego pose at this timestamp
        lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_data  = self.nusc.get("sample_data", lidar_token)
        ego_pose    = self.nusc.get("ego_pose", lidar_data["ego_pose_token"])

        # global -> ego transform
        ego_rot   = Quaternion(ego_pose["rotation"]).inverse
        ego_trans = np.array(ego_pose["translation"])

        boxes  = []
        labels = []

        for ann_token in sample["anns"]:
            ann = self.nusc.get("sample_annotation", ann_token)

            # map nuScenes category to detection class index
            category  = ann["category_name"]
            class_idx = self._map_category(category)

            # skip unknown categories
            if class_idx < 0:
                continue

            # annotation center in global frame
            center_global = np.array(ann["translation"])  # (3,)

            # transform center to ego frame
            center_ego = ego_rot.rotate(center_global - ego_trans)

            # annotation dimensions: nuScenes gives (w, l, h) -> we use (l, w, h)
            w, l, h = ann["size"]

            # annotation rotation in global frame -> ego frame
            ann_rot    = Quaternion(ann["rotation"])
            rot_ego    = ego_rot * ann_rot
            yaw        = rot_ego.yaw_pitch_roll[0]  # yaw only for BEV detection

            boxes.append([
                center_ego[0], center_ego[1], center_ego[2],
                l, w, h, yaw,
            ])
            labels.append(class_idx)

        if len(boxes) == 0:
            return np.zeros((0, 7), dtype=np.float32), np.zeros((0,), dtype=np.int64)

        return (
            np.array(boxes,  dtype=np.float32),
            np.array(labels, dtype=np.int64),
        )

    def _map_category(self, category_name: str) -> int:
        """
        Maps a nuScenes category name to a detection class index.

        nuScenes has hierarchical category names like 'vehicle.car',
        'human.pedestrian.adult', etc. We map them to the 10 CenterPoint
        classes by matching the fine-grained name to the coarse class.

        Returns -1 for categories not in DET_CLASSES (ignored during training).
        """
        mapping = {
            "vehicle.car"                    : "car",
            "vehicle.truck"                  : "truck",
            "vehicle.construction"           : "construction_vehicle",
            "vehicle.bus.bendy"              : "bus",
            "vehicle.bus.rigid"              : "bus",
            "vehicle.trailer"                : "trailer",
            "vehicle.motorcycle"             : "motorcycle",
            "vehicle.bicycle"                : "bicycle",
            "human.pedestrian.adult"         : "pedestrian",
            "human.pedestrian.child"         : "pedestrian",
            "human.pedestrian.wheelchair"    : "pedestrian",
            "human.pedestrian.stroller"      : "pedestrian",
            "human.pedestrian.personal_mobility" : "pedestrian",
            "human.pedestrian.police_officer": "pedestrian",
            "human.pedestrian.construction_worker" : "pedestrian",
            "movable_object.barrier"         : "barrier",
            "movable_object.trafficcone"     : "traffic_cone",
        }

        coarse = mapping.get(category_name, None)
        if coarse is None:
            return -1
        return self.class_to_idx.get(coarse, -1)