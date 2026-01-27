import os, json, pickle, tqdm
import torch, cv2
from torch.utils.data import Dataset
from pyquaternion import Quaternion
import numpy as np
import open3d as o3d

from dust3r.utils.image import imread_cv2
from driv3r.datasets.base_many_view_dataset import BaseManyViewDataset

class NuSceneDataset(BaseManyViewDataset):

    def __init__(self, 
                 data_root,
                 sequence_length,
                 cams, 
                 depth_root,
                 depth_mode="r3d3",
                 dynamic=False,
                 dynamic_metas=None,
                 *args,
                 **kwargs):

        super().__init__(*args, **kwargs)
        self.data_root = data_root
        self.sequence_length = sequence_length
        self.cams = cams
        self.depth_root = depth_root
        self.depth_mode = depth_mode
        self.dynamic = dynamic
        self.dynamic_metas = dynamic_metas
        self.dynamic_scene_list = []

        if self.split == 'train':
            with open(os.path.join(self.data_root, "train_metas.pkl"), 'rb') as f:
                self.metas = pickle.load(f)
        elif self.split == 'val':
            with open(os.path.join(self.data_root, "val_metas.pkl"), 'rb') as f:
                self.metas = pickle.load(f)
        else:
            raise NotImplementedError
        
        self.scene_list = self.load_scenes()

        if self.dynamic:
            with open(self.dynamic_metas, 'rb') as f:
                self.dynamic_meta_list = pickle.load(f)
                sequence_ids = list(self.dynamic_meta_list.keys())

                for sequence_id in sequence_ids:
                    sequence_infos = self.scene_list[sequence_id]
                    dynamic_sequence_infos = []
                    for sequence_info in sequence_infos:
                        sequence_info['splits'] = self.dynamic_meta_list[sequence_id]
                        dynamic_sequence_infos.append(sequence_info)
                    self.dynamic_scene_list.append(dynamic_sequence_infos)

    def load_scenes(self):
        
        sequence_length = self.sequence_length
        scene_list = []

        for scene_token, scene_metas in self.metas.items():
            # sort by timestamp
            scene_metas = dict(sorted(scene_metas.items(), key=lambda item: item[1]["CAM_FRONT"]["timestamp"]))
            sample_token_list = [sample_token for sample_token in scene_metas.keys()]

            for video_idx in range(len(scene_metas) // sequence_length):
                # sample token for current sequence
                sample_tokens = sample_token_list[video_idx * sequence_length : (video_idx + 1) * sequence_length]
                for cam in self.cams:
                    video_metas = []
                    for token in sample_tokens:
                        video_metas.append(dict(
                            timestamp=scene_metas[token][cam]["timestamp"],
                            img=os.path.join(self.data_root, scene_metas[token][cam]["filename"]),
                            pcd=os.path.join(self.data_root, scene_metas[token]["LIDAR_TOP"]["filename"]),
                            ego_pose=scene_metas[token][cam]["ego_pose"],
                            lidar_pose=scene_metas[token]["LIDAR_TOP"]["calibrated_sensor"],
                            cam_poses=scene_metas[token][cam]["calibrated_sensor"],
                            image_wh=scene_metas[token][cam]["image_wh"]
                        ))
                    # input metas
                    sequence_metas = []
                    for sample_idx in range(sequence_length):

                        cur_sample_meta = video_metas[sample_idx]
                        cam_poses = cur_sample_meta["cam_poses"]
                        e2w = cur_sample_meta["ego_pose"]
                        l2e = cur_sample_meta["lidar_pose"]
                        c2e_matrixs = NuSceneDataset.get_extrinsic_matrix(
                            rot=cam_poses["rotation"],
                            trans=cam_poses["translation"]
                        )
                        l2e_matrix = NuSceneDataset.get_extrinsic_matrix(
                            rot=l2e["rotation"],
                            trans=l2e["translation"]
                        )
                        e2w_matrix = NuSceneDataset.get_extrinsic_matrix(
                            rot=e2w["rotation"],
                            trans=e2w["translation"]
                        )

                        sequence_metas.append(dict(
                            timestamp=cur_sample_meta["timestamp"],
                            img=cur_sample_meta["img"],
                            pcd=cur_sample_meta["pcd"],
                            camera_intrinsics=np.array(cam_poses["intrinsics"]),
                            lidar_pose=e2w_matrix @ l2e_matrix,
                            cam_poses=e2w_matrix @ c2e_matrixs,
                            true_shape=cur_sample_meta["image_wh"]
                        ))

                    scene_list.append(sequence_metas)
            
        return scene_list

    def __len__(self):
        if self.dynamic:
            return len(self.dynamic_scene_list)
        else:
            return len(self.scene_list)

    @staticmethod
    def get_extrinsic_matrix(rot, trans):
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = Quaternion(rot).rotation_matrix
        extrinsic[:3, 3] = np.array(trans)
        return extrinsic

    @staticmethod
    def load_lidar_pts(views, image_size=(224, 224)):

        h, w = image_size
        device = views[0]['img'].device

        new_views = []
        # Load Lidar points for evaluation, assert batch_size = 1
        for view in views:

            pts3d = torch.zeros((1, h, w, 3), dtype=torch.float32, device=device)
            valid_mask = torch.zeros((1, h, w), dtype=torch.bool, device=device)

            pcd_l = np.fromfile(view["pcd"][0], dtype=np.float32)
            pcd_l = pcd_l.reshape(-1, 5)[:, :3]
            if pcd_l.size == 0:
                view['pts3d'] = pts3d
                view['valid_mask'] = valid_mask
                new_views.append(view)
                continue

            N = pcd_l.shape[0]
            ones = np.ones((N, 1), dtype=np.float32)
            pcd_l_homo = np.concatenate((pcd_l, ones), axis=-1)

            l2w_matrix = view["lidar_pose"][0].cpu().numpy()
            c2w_matrix = view["camera_pose"][0].cpu().numpy()
            l2c_matrix = np.linalg.inv(c2w_matrix) @ l2w_matrix

            pcd_c_homo = (l2c_matrix @ pcd_l_homo.T).T
            z = pcd_c_homo[:, 2]

            K3 = view['camera_intrinsics']
            if isinstance(K3, torch.Tensor):
                K3 = K3.cpu().numpy()
            if K3.ndim == 3:
                K3 = K3[0]
            fx, fy = float(K3[0, 0]), float(K3[1, 1])
            cx, cy = float(K3[0, 2]), float(K3[1, 2])

            z_valid = np.isfinite(z) & (z > 1e-6)
            if not np.any(z_valid):
                view['pts3d'] = pts3d
                view['valid_mask'] = valid_mask
                new_views.append(view)
                continue

            pcd_c = pcd_c_homo[z_valid]
            z = z[z_valid]
            u = (pcd_c[:, 0] * fx / z + cx).astype(np.int32)
            v = (pcd_c[:, 1] * fy / z + cy).astype(np.int32)

            xy_valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            u = u[xy_valid]
            v = v[xy_valid]
            z = z[xy_valid]

            if z.size > 0:
                idx = v * w + u
                depth = np.full((h * w,), np.inf, dtype=np.float32)
                np.minimum.at(depth, idx, z.astype(np.float32))
                depth = depth.reshape(h, w)
                mask = np.isfinite(depth)

                ys, xs = np.where(mask)
                depth_vals = depth[ys, xs]

                x_cam = (xs - cx) * depth_vals / fx
                y_cam = (ys - cy) * depth_vals / fy
                pts_cam = np.stack([x_cam, y_cam, depth_vals, np.ones_like(depth_vals)], axis=1)
                pts_world = (c2w_matrix @ pts_cam.T).T[:, :3]

                ys_t = torch.tensor(ys, dtype=torch.long, device=device)
                xs_t = torch.tensor(xs, dtype=torch.long, device=device)
                pts3d[:, ys_t, xs_t, :] = torch.tensor(pts_world, dtype=torch.float32, device=device)
                valid_mask[:, ys_t, xs_t] = True

            view['pts3d'] = pts3d
            view['valid_mask'] = valid_mask
            new_views.append(view)
        
        return new_views

    def get_depth_map_via_lidar(self, sample_meta, image_size):
        
        h, w, _ = image_size
        pcd_l = np.fromfile(sample_meta["pcd"], dtype=np.float32)
        pcd_l = pcd_l.reshape(-1, 5)[:, :3]
        if pcd_l.size == 0:
            return np.zeros((h, w), dtype=np.float32)

        N = pcd_l.shape[0]
        ones = np.ones((N, 1), dtype=np.float32)
        pcd_l_homo = np.concatenate((pcd_l, ones), axis=-1)

        l2w_matrix = sample_meta["lidar_pose"]
        c2w_matrix = sample_meta["camera_pose"]
        l2c_matrix = np.linalg.inv(c2w_matrix) @ l2w_matrix

        pcd_c_homo = (l2c_matrix @ pcd_l_homo.T).T

        z = pcd_c_homo[:, 2]
        K3 = sample_meta['camera_intrinsics']
        fx, fy = float(K3[0, 0]), float(K3[1, 1])
        cx, cy = float(K3[0, 2]), float(K3[1, 2])

        z_valid = np.isfinite(z) & (z > 1e-6)
        if not np.any(z_valid):
            return np.zeros((h, w), dtype=np.float32)

        pcd_c = pcd_c_homo[z_valid]
        z = z[z_valid]
        u = (pcd_c[:, 0] * fx / z + cx).astype(np.int32)
        v = (pcd_c[:, 1] * fy / z + cy).astype(np.int32)

        valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u = u[valid]
        v = v[valid]
        z = z[valid]

        depth = np.full((h * w,), np.inf, dtype=np.float32)
        if z.size > 0:
            idx = v * w + u
            np.minimum.at(depth, idx, z.astype(np.float32))
        depth = depth.reshape(h, w)
        depth[np.isinf(depth)] = 0.0
        return depth
    
    def get_r3d3_depth_map(self, 
                           meta,
                           true_image_shape, 
                           resolution,
                           nus_intri):
        
        h_nus, w_nus, _ = true_image_shape
        rgb_filename = os.path.basename(meta["img"])
        cam = rgb_filename.split("__")[1]
        r3d3_depth_path = os.path.join(
            self.depth_root, cam,
            rgb_filename.replace('.jpg', '_depth(0)_pred.npz')
        )
        assert os.path.exists(r3d3_depth_path)
        
        r3d3_npz = np.load(r3d3_depth_path)
        r3d3_depth = r3d3_npz['depth']
        
        r3d3_depth = np.pad(r3d3_depth, ((1, 1), (16, 16)), mode='constant', constant_values=0)
        r3d3_depth = cv2.resize(r3d3_depth, (w_nus, h_nus), interpolation=cv2.INTER_LANCZOS4)
   
        return r3d3_depth
        
    def _get_views(self, idx, resolution, rng):
        
        views = []
        if self.dynamic:
            video_frames = self.dynamic_scene_list[idx]
        else:
            video_frames = self.scene_list[idx]
    

        for frame_meta in video_frames:
            rgb_image = imread_cv2(frame_meta["img"])
            intrinsics = frame_meta['camera_intrinsics'].astype(np.float32)
            frame_meta_local = dict(frame_meta)
            frame_meta_local["camera_pose"] = frame_meta["cam_poses"]

            if self.depth_mode in ("none", "zeros"):
                # Depthmap is not used by Driv3R inference and is often overridden by load_lidar_pts()
                # during evaluation/visualization. Returning zeros here avoids expensive lidar projection
                # or missing R3D3 files when we only need images + poses.
                depthmap = np.zeros(rgb_image.shape[:2], dtype=np.float32)
            elif self.depth_mode == "r3d3":
                depthmap = self.get_r3d3_depth_map(
                    meta=frame_meta, 
                    true_image_shape=rgb_image.shape, 
                    resolution=resolution,
                    nus_intri=intrinsics
                )
            elif self.depth_mode == "lidar":
                depthmap = self.get_depth_map_via_lidar(frame_meta_local, rgb_image.shape)
            else:
                raise NotImplementedError(f"Unsupported depth_mode: {self.depth_mode}")
            
            # split into two frames, left and right
            if self.dynamic:
                splits = frame_meta['splits']
            else:
                splits = ['left', 'right']
            for split in splits:
                split_rgb_image, split_depthmap, split_intrinsics = self._crop_resize_if_necessary(
                    rgb_image, depthmap, intrinsics, resolution, 
                    rng=rng, info=frame_meta["img"], split=split
                )
            
                views.append(dict(
                    img=split_rgb_image,
                    depthmap=split_depthmap,
                    camera_pose=frame_meta['cam_poses'].astype(np.float32),
                    pcd=frame_meta['pcd'],
                    camera_intrinsics=split_intrinsics,
                    lidar_pose=frame_meta['lidar_pose'].astype(np.float32),
                    dataset='nuscenes',
                    label=frame_meta["img"],
                    instance=os.path.basename(frame_meta["img"])
                ))
        
        return views


if __name__ == '__main__':

    cams=[
        'CAM_FRONT',
        'CAM_FRONT_LEFT',
        'CAM_FRONT_RIGHT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_BACK_RIGHT'
    ]

    dataset = NuSceneDataset(
        data_root='datasets/nuscenes',
        sequence_length=5,
        cams=cams,
        split='val',
        depth_root='third_party/r3d3/pred/samples',
        resolution=224
    )
