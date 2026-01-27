import os
import yaml
import pickle
import numpy as np
import cv2
import torch
import open3d as o3d

from dust3r.utils.image import imread_cv2
from driv3r.datasets.base_many_view_dataset import BaseManyViewDataset


def _x_to_world(pose):
    """Pose -> 4x4 matrix. pose = [x, y, z, roll, yaw, pitch] in degrees."""
    x, y, z, roll, yaw, pitch = pose[:]
    c_y = np.cos(np.radians(yaw))
    s_y = np.sin(np.radians(yaw))
    c_r = np.cos(np.radians(roll))
    s_r = np.sin(np.radians(roll))
    c_p = np.cos(np.radians(pitch))
    s_p = np.sin(np.radians(pitch))

    matrix = np.identity(4, dtype=np.float32)
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z

    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r
    return matrix


def _x1_to_x2(x1, x2):
    x1_to_world = _x_to_world(x1)
    x2_to_world = _x_to_world(x2)
    world_to_x2 = np.linalg.inv(x2_to_world)
    return world_to_x2 @ x1_to_world


def _read_pcd_xyz(pcd_path):
    pcd = o3d.io.read_point_cloud(pcd_path)
    xyz = np.asarray(pcd.points, dtype=np.float32)
    return xyz


UE4_FROM_OPENCV = np.array(
    [[0, 0, 1, 0],
     [1, 0, 0, 0],
     [0, -1, 0, 0],
     [0, 0, 0, 1]],
    dtype=np.float32
)


class OPV2VDataset(BaseManyViewDataset):
    def __init__(self,
                 data_root,
                 sequence_length,
                 cams,
                 split,
                 cav_mode="ego",
                 max_cav=5,
                 split_mode="both",
                 max_scenes=None,
                 depth_mode="lidar",
                 depth_root=None,
                 cache=True,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.data_root = data_root
        self.sequence_length = sequence_length
        self.cams = [self._normalize_cam_name(c) for c in cams]
        self.split = split
        self.cav_mode = cav_mode
        self.max_cav = max_cav
        self.split_mode = split_mode
        self.max_scenes = max_scenes
        self.depth_mode = depth_mode
        self.depth_root = depth_root
        self.cache = cache

        self.scene_list = self.load_scenes()

    @staticmethod
    def _normalize_cam_name(cam):
        if isinstance(cam, int):
            return f"camera{cam}"
        cam = str(cam)
        if cam.startswith("camera"):
            return cam
        if cam.isdigit():
            return f"camera{cam}"
        return cam

    @staticmethod
    def _resolve_split(split):
        split = split.lower()
        if split in ("val", "valid", "validation"):
            return "validate"
        return split

    def _select_cavs(self, cav_list):
        cav_list = sorted(cav_list, key=lambda x: int(x))
        # move RSU (negative ids) to the end
        if cav_list and int(cav_list[0]) < 0:
            cav_list = cav_list[1:] + [cav_list[0]]
        if self.cav_mode == "ego":
            return cav_list[:1]
        return cav_list[: self.max_cav]

    def load_scenes(self):
        split_dir = self._resolve_split(self.split)
        root_dir = os.path.join(self.data_root, split_dir)
        cache_key = f"opv2v_metas_{split_dir}_{self.cav_mode}"
        if self.cav_mode != "ego":
            cache_key += f"_maxcav{self.max_cav}"
        cache_key += f"_len{self.sequence_length}_cams{len(self.cams)}"
        if self.max_scenes is not None:
            cache_key += f"_maxscenes{int(self.max_scenes)}"
        cache_path = os.path.join(self.data_root, f"{cache_key}.pkl")
        if self.cache:
            if os.path.isfile(cache_path):
                with open(cache_path, "rb") as f:
                    return pickle.load(f)
            # Backward-compatibility: older cache filenames did not include `_maxcav{}`.
            # Reuse them if present to avoid very slow YAML scanning on first run.
            if self.cav_mode != "ego":
                legacy_key = f"opv2v_metas_{split_dir}_{self.cav_mode}"
                legacy_key += f"_len{self.sequence_length}_cams{len(self.cams)}"
                if self.max_scenes is not None:
                    legacy_key += f"_maxscenes{int(self.max_scenes)}"
                legacy_path = os.path.join(self.data_root, f"{legacy_key}.pkl")
                if os.path.isfile(legacy_path):
                    with open(legacy_path, "rb") as f:
                        return pickle.load(f)

        scenario_folders = sorted([
            os.path.join(root_dir, x)
            for x in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, x))
        ])

        scene_list = []
        stop = False
        for scenario_folder in scenario_folders:
            cav_list = [x for x in os.listdir(scenario_folder)
                        if os.path.isdir(os.path.join(scenario_folder, x))]
            if not cav_list:
                continue

            cav_list = self._select_cavs(cav_list)

            for cav_id in cav_list:
                cav_path = os.path.join(scenario_folder, cav_id)
                yaml_files = sorted([
                    os.path.join(cav_path, x)
                    for x in os.listdir(cav_path)
                    if x.endswith(".yaml") and "additional" not in x
                ])

                # match OpenCOOD filter
                yaml_files = [x for x in yaml_files if not ("2021_08_20_21_10_24" in x and "000265" in x)]

                timestamps = [os.path.basename(x).replace(".yaml", "") for x in yaml_files]
                if len(timestamps) < self.sequence_length:
                    continue

                # cache params on-demand (parsing all YAMLs upfront is very slow)
                params_cache = {}

                for video_idx in range(len(timestamps) // self.sequence_length):
                    seq_timestamps = timestamps[
                        video_idx * self.sequence_length:
                        (video_idx + 1) * self.sequence_length
                    ]
                    for cam in self.cams:
                        sequence_metas = []
                        for ts in seq_timestamps:
                            params = params_cache.get(ts)
                            if params is None:
                                yaml_path = os.path.join(cav_path, f"{ts}.yaml")
                                with open(yaml_path, "r") as f:
                                    try:
                                        params = yaml.safe_load(f)
                                    except yaml.YAMLError:
                                        f.seek(0)
                                        # Some OPV2V YAMLs contain numpy tags; fall back to unsafe loader.
                                        params = yaml.load(f, Loader=yaml.UnsafeLoader)
                                params_cache[ts] = params

                            img_path = os.path.join(cav_path, f"{ts}_{cam}.png")
                            pcd_path = os.path.join(cav_path, f"{ts}.pcd")

                            lidar_pose = _x_to_world(params["lidar_pose"]).astype(np.float32)
                            cam_to_lidar = _x1_to_x2(params[cam]["cords"], params["lidar_pose"])
                            cam_to_lidar = cam_to_lidar @ UE4_FROM_OPENCV
                            cam_pose = (lidar_pose @ cam_to_lidar).astype(np.float32)
                            intrinsics = np.array(params[cam]["intrinsic"], dtype=np.float32)

                            sequence_metas.append(dict(
                                timestamp=ts,
                                img=img_path,
                                pcd=pcd_path,
                                camera_intrinsics=intrinsics,
                                cam_poses=cam_pose,
                                lidar_pose=lidar_pose,
                                scenario=os.path.basename(scenario_folder),
                                cav_id=cav_id,
                                cam=cam,
                            ))
                        scene_list.append(sequence_metas)
                        if self.max_scenes is not None and len(scene_list) >= int(self.max_scenes):
                            stop = True
                            break
                    if stop:
                        break
                if stop:
                    break
            if stop:
                break
        if self.cache:
            with open(cache_path, "wb") as f:
                pickle.dump(scene_list, f)
        return scene_list

    def __len__(self):
        return len(self.scene_list)

    def _get_depth_map_via_lidar(self, sample_meta, image_size):
        h, w, _ = image_size
        pcd_xyz = _read_pcd_xyz(sample_meta["pcd"])
        if pcd_xyz.size == 0:
            return np.zeros((h, w), dtype=np.float32)

        ones = np.ones((pcd_xyz.shape[0], 1), dtype=np.float32)
        pcd_h = np.concatenate([pcd_xyz, ones], axis=1)

        l2w = sample_meta["lidar_pose"]
        c2w = sample_meta["camera_pose"]
        l2c = np.linalg.inv(c2w) @ l2w

        pcd_c = (l2c @ pcd_h.T).T
        z = pcd_c[:, 2]

        K = np.eye(4, dtype=np.float32)
        K[:3, :3] = sample_meta["camera_intrinsics"]
        pcd_pixel = (K @ pcd_c.T).T
        denom = pcd_pixel[:, 2] + 1e-7
        x = pcd_pixel[:, 0] / denom
        y = pcd_pixel[:, 1] / denom

        valid = (z > 0) & np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x < w) & (y >= 0) & (y < h)
        x = x[valid].astype(np.int32)
        y = y[valid].astype(np.int32)
        z = z[valid]

        depth_map = np.full((h, w), np.inf, dtype=np.float32)
        if z.size > 0:
            idx = y * w + x
            depth_flat = depth_map.reshape(-1)
            np.minimum.at(depth_flat, idx, z)
            depth_map = depth_flat.reshape(h, w)
        depth_map[np.isinf(depth_map)] = 0.0
        return depth_map

    def _get_views(self, idx, resolution, rng):
        views = []
        video_frames = self.scene_list[idx]

        for frame_meta in video_frames:
            rgb_image = imread_cv2(frame_meta["img"])
            intrinsics = frame_meta["camera_intrinsics"].astype(np.float32)

            frame_meta_local = dict(frame_meta)
            frame_meta_local["camera_pose"] = frame_meta["cam_poses"].astype(np.float32)

            if self.depth_mode == "lidar":
                depthmap = self._get_depth_map_via_lidar(frame_meta_local, rgb_image.shape)
            else:
                raise NotImplementedError("Only lidar depth is supported for OPV2V.")

            split_mode = str(self.split_mode).lower()
            if split_mode in ("both", "lr", "left_right"):
                splits = ["left", "right"]
            elif split_mode in ("left", "l"):
                splits = ["left"]
            elif split_mode in ("right", "r"):
                splits = ["right"]
            elif split_mode in ("random", "rand"):
                splits = [rng.choice(["left", "right"])]
            else:
                raise ValueError(f"Unsupported split_mode: {self.split_mode}")

            for split in splits:
                split_rgb_image, split_depthmap, split_intrinsics = self._crop_resize_if_necessary(
                    rgb_image, depthmap, intrinsics, resolution,
                    rng=rng, info=frame_meta["img"], split=split
                )

                views.append(dict(
                    img=split_rgb_image,
                    depthmap=split_depthmap,
                    camera_pose=frame_meta_local["camera_pose"].astype(np.float32),
                    pcd=frame_meta["pcd"],
                    camera_intrinsics=split_intrinsics,
                    lidar_pose=frame_meta["lidar_pose"].astype(np.float32),
                    dataset="opv2v",
                    label=f"{frame_meta['scenario']}_{frame_meta['cav_id']}_{frame_meta['cam']}",
                    instance=f"{frame_meta['timestamp']}"
                ))

        return views

    @staticmethod
    def load_lidar_pts(views, image_size=(224, 224)):
        h, w = image_size
        device = views[0]["img"].device

        new_views = []
        for view in views:
            pts3d = torch.zeros((1, h, w, 3), dtype=torch.float32, device=device)
            valid_mask = torch.zeros((1, h, w), dtype=torch.bool, device=device)

            pcd_xyz = _read_pcd_xyz(view["pcd"][0])
            if pcd_xyz.size == 0:
                view["pts3d"] = pts3d
                view["valid_mask"] = valid_mask
                new_views.append(view)
                continue

            ones = np.ones((pcd_xyz.shape[0], 1), dtype=np.float32)
            pcd_h = np.concatenate([pcd_xyz, ones], axis=1)

            l2w = view["lidar_pose"][0].cpu().numpy()
            c2w = view["camera_pose"][0].cpu().numpy()
            l2c = np.linalg.inv(c2w) @ l2w

            pcd_c = (l2c @ pcd_h.T).T
            z = pcd_c[:, 2]

            K = np.eye(4, dtype=np.float32)
            K3 = view["camera_intrinsics"].cpu().numpy()
            if K3.ndim == 3:
                K3 = K3[0]
            K[:3, :3] = K3
            pcd_pixel = (K @ pcd_c.T).T
            denom = pcd_pixel[:, 2] + 1e-7
            x = pcd_pixel[:, 0] / denom
            y = pcd_pixel[:, 1] / denom

            valid = (z > 0) & np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x < w) & (y >= 0) & (y < h)
            x = x[valid].astype(np.int32)
            y = y[valid].astype(np.int32)
            z = z[valid]

            if z.size > 0:
                idx = y * w + x
                # keep nearest point per pixel
                depth = np.full((h * w,), np.inf, dtype=np.float32)
                np.minimum.at(depth, idx, z)
                depth = depth.reshape(h, w)
                mask = np.isfinite(depth)

                # reconstruct world points for valid pixels
                ys, xs = np.where(mask)
                depth_vals = depth[ys, xs]
                # back-project to camera
                K3 = view["camera_intrinsics"].cpu().numpy()
                if K3.ndim == 3:
                    K3 = K3[0]
                fx, fy = K3[0, 0], K3[1, 1]
                cx, cy = K3[0, 2], K3[1, 2]
                x_cam = (xs - cx) * depth_vals / fx
                y_cam = (ys - cy) * depth_vals / fy
                pts_cam = np.stack([x_cam, y_cam, depth_vals, np.ones_like(depth_vals)], axis=1)
                pts_world = (c2w @ pts_cam.T).T[:, :3]

                valid_pcd_pixel = torch.tensor(np.stack([xs, ys], axis=1), dtype=torch.long, device=device)
                y_indices = valid_pcd_pixel[:, 1]
                x_indices = valid_pcd_pixel[:, 0]
                pts3d[:, y_indices, x_indices, :] = torch.tensor(pts_world, dtype=torch.float32, device=device)
                valid_mask[:, y_indices, x_indices] = True

            view["pts3d"] = pts3d
            view["valid_mask"] = valid_mask
            new_views.append(view)

        return new_views
