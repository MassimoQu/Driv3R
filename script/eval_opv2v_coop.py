import argparse
import copy
import json
import os
from pathlib import Path
import sys

# Allow running as `python3 script/eval_opv2v_coop.py` from anywhere.
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import numpy as np
import open3d as o3d
import torch
import yaml
from PIL import Image

from dust3r.datasets.utils.transforms import ImgNorm
from dust3r.losses import L21
from dust3r.utils.geometry import geotrf
from dust3r.utils.image import imread_cv2
from driv3r.datasets.opv2v import UE4_FROM_OPENCV, _read_pcd_xyz, _x1_to_x2, _x_to_world
from driv3r.loss import Regr3D_t_ScaleShiftInv
from driv3r.model import Spann3R
from driv3r.tools.eval_recon import accuracy, completion, completion_ratio


def _load_yaml(path: Path) -> dict:
    with path.open("r") as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError:
            f.seek(0)
            return yaml.load(f, Loader=yaml.UnsafeLoader)


def _select_cavs(cav_list, max_cav: int):
    cav_list = sorted(cav_list, key=lambda x: int(x))
    if cav_list and int(cav_list[0]) < 0:
        cav_list = cav_list[1:] + [cav_list[0]]
    return cav_list[: max(1, int(max_cav))]


def _crop_resize_if_necessary(image_np, intrinsics: np.ndarray, resolution: int, split: str):
    # Mirrors dust3r BaseStereoViewDataset._crop_resize_if_necessary for square output.
    image = Image.fromarray(image_np) if not isinstance(image_np, Image.Image) else image_np
    w, h = image.size
    output_width = int(resolution)
    output_height = int(resolution)
    scaling_factor = output_height / float(h)
    target_height = output_height
    target_width = int(w * scaling_factor)

    image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)

    fx = float(intrinsics[0, 0]) * scaling_factor
    fy = float(intrinsics[1, 1]) * scaling_factor
    cx = float(intrinsics[0, 2]) * scaling_factor
    cy = float(intrinsics[1, 2]) * scaling_factor

    split = str(split).lower()
    if split in ("left", "l"):
        image = image.crop((0, 0, output_width, output_height))
        intrinsics2 = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        return image, intrinsics2

    if split in ("right", "r"):
        image = image.crop((target_width - output_width, 0, target_width, output_height))
        crop_cx = target_width - 0.5 * output_width
        cx = cx + (output_width - 1) / 2 - crop_cx
        intrinsics2 = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        return image, intrinsics2

    raise ValueError(f"Unsupported split: {split}")


def _build_views_for_sequence(
    cav_path: Path,
    timestamps,
    cam: str,
    resolution: int,
    split: str,
):
    views = []
    for ts in timestamps:
        yaml_path = cav_path / f"{ts}.yaml"
        img_path = cav_path / f"{ts}_{cam}.png"
        pcd_path = cav_path / f"{ts}.pcd"

        params = _load_yaml(yaml_path)
        rgb = imread_cv2(str(img_path))

        lidar_pose = _x_to_world(params["lidar_pose"]).astype(np.float32)
        cam_to_lidar = _x1_to_x2(params[cam]["cords"], params["lidar_pose"]).astype(np.float32)
        cam_to_lidar = cam_to_lidar @ UE4_FROM_OPENCV
        cam_pose = (lidar_pose @ cam_to_lidar).astype(np.float32)
        intrinsics = np.array(params[cam]["intrinsic"], dtype=np.float32)

        rgb_pil, intrinsics2 = _crop_resize_if_necessary(rgb, intrinsics, resolution, split)

        img_t = ImgNorm(rgb_pil).unsqueeze(0)  # (1,3,H,W)

        views.append(
            dict(
                img=img_t,
                camera_pose=torch.from_numpy(cam_pose).unsqueeze(0),  # (1,4,4)
                camera_intrinsics=torch.from_numpy(intrinsics2).unsqueeze(0),  # (1,3,3)
                lidar_pose=torch.from_numpy(lidar_pose).unsqueeze(0),  # (1,4,4)
                pcd=[str(pcd_path)],
                dataset="opv2v",
                instance=str(ts),
                label=f"{cav_path.parent.name}_{cav_path.name}_{cam}",
            )
        )
    return views


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _pcd_from(points_xyz, colors_rgb=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz.astype(np.float64))
    if colors_rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors_rgb.astype(np.float64))
    return pcd


def _apply_outlier_removal(
    pcd: o3d.geometry.PointCloud,
    *,
    stat_nb: int,
    stat_std: float,
    radius_nb: int,
    radius: float,
) -> o3d.geometry.PointCloud:
    if pcd is None or len(pcd.points) == 0:
        return pcd
    out = pcd
    if stat_nb and int(stat_nb) > 0:
        out, _ = out.remove_statistical_outlier(nb_neighbors=int(stat_nb), std_ratio=float(stat_std))
    if radius_nb and int(radius_nb) > 0 and radius and float(radius) > 0:
        out, _ = out.remove_radius_outlier(nb_points=int(radius_nb), radius=float(radius))
    return out


def _rot_angle_deg(R: np.ndarray) -> float:
    tr = float(np.trace(R[:3, :3]))
    c = (tr - 1.0) / 2.0
    c = float(np.clip(c, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def _centered_icp_t_norm(T: np.ndarray, pred_center: np.ndarray, gt_center: np.ndarray) -> float:
    # Measure translation after expressing the rigid transform between *centered* frames.
    # pred_centered = pred - pred_center
    # gt_centered   = gt   - gt_center
    # Then: gt_centered ≈ Tc @ pred_centered, where Tc = Trans(-gt_center) @ T @ Trans(pred_center).
    A = np.eye(4, dtype=np.float64)
    A[:3, 3] = pred_center.reshape(3)
    B = np.eye(4, dtype=np.float64)
    B[:3, 3] = -gt_center.reshape(3)
    Tc = B @ T @ A
    return float(np.linalg.norm(Tc[:3, 3]))


def _mask_pred(
    pts_cam1: np.ndarray,
    *,
    conf: np.ndarray | None,
    conf_thresh: float | None,
    conf_keep: str = "ge",
    conf_quantile: float | None = None,
    depth_min: float | None,
    depth_max: float | None,
    lidar_mask: np.ndarray | None = None,
    drop_mask: np.ndarray | None = None,
    sky_top_ratio: float | None = None,
):
    finite = np.isfinite(pts_cam1).all(axis=-1)
    keep = finite.copy()

    if lidar_mask is not None:
        keep &= lidar_mask.astype(bool)

    if depth_min is not None or depth_max is not None:
        z = pts_cam1[..., 2]
        if depth_min is None:
            depth_min = -np.inf
        if depth_max is None:
            depth_max = np.inf
        keep &= (z >= float(depth_min)) & (z <= float(depth_max))

    if conf is not None:
        if conf_quantile is not None:
            q = float(conf_quantile)
            if not (0.0 <= q <= 1.0):
                raise ValueError(f"--conf_quantile must be in [0,1], got: {conf_quantile}")
            finite_conf = np.isfinite(conf)
            if np.any(finite_conf):
                thr = float(np.quantile(conf[finite_conf], q))
                keep &= conf >= thr
        elif conf_thresh is not None:
            if str(conf_keep).lower() in ("ge", "gte", ">=", "high"):
                keep &= conf >= float(conf_thresh)
            elif str(conf_keep).lower() in ("le", "lte", "<=", "low"):
                keep &= conf <= float(conf_thresh)
            else:
                raise ValueError(f"Unsupported conf_keep: {conf_keep} (expected 'ge' or 'le')")

    if drop_mask is not None:
        keep &= ~drop_mask.astype(bool)

    if sky_top_ratio is not None and float(sky_top_ratio) > 0:
        h = int(pts_cam1.shape[0])
        w = int(pts_cam1.shape[1])
        sky_h = int(round(h * float(sky_top_ratio)))
        if sky_h > 0:
            sky = np.zeros((h, w), dtype=bool)
            sky[:sky_h, :] = True
            keep &= ~sky

    return keep


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser("OPV2V cooperative evaluation (fuse multiple CAVs by union of point clouds).")
    ap.add_argument("--ckpt_path", type=str, required=True)
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--split", type=str, default="validate")
    ap.add_argument("--mode", choices=["single", "coop"], required=True)
    ap.add_argument("--max_cav", type=int, default=5, help="For coop mode, number of CAVs to fuse (>=1).")
    ap.add_argument(
        "--camera",
        type=str,
        default="camera0",
        help="Camera(s) to use: camera0..camera3, a comma-separated list, or 'all'.",
    )
    ap.add_argument(
        "--split_view",
        type=str,
        default="left",
        choices=["left", "right", "both"],
        help="Crop view: left/right, or 'both' to run both crops and union the resulting point clouds (closer to nuScenes eval).",
    )
    ap.add_argument("--resolution", type=int, default=224)
    ap.add_argument("--sequence_length", type=int, default=5)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--max_scenes", type=int, default=50)
    ap.add_argument("--icp_threshold", type=float, default=100.0)
    ap.add_argument("--voxel_size", type=float, default=0.2, help="Downsample before ICP/metrics; 0 disables.")
    ap.add_argument(
        "--mem_sim_thresh",
        type=float,
        default=None,
        help="Spann3R eval-time memory similarity threshold. 1.0 disables similarity skipping (often better for OPV2V).",
    )
    ap.add_argument(
        "--vis_voxel_size",
        type=float,
        default=0.05,
        help="Downsample before saving PLYs (for faster web viewing); 0 disables.",
    )
    ap.add_argument(
        "--conf_thresh",
        type=float,
        default=1.0,
        help="Confidence threshold (see --conf_keep). For 'ge', set 1.0 to effectively disable (conf>=1).",
    )
    ap.add_argument(
        "--conf_quantile",
        type=float,
        default=None,
        help="If set in [0,1], overrides --conf_thresh and keeps conf >= quantile(conf, q) per frame (high-conf filtering).",
    )
    ap.add_argument(
        "--conf_keep",
        type=str,
        default="ge",
        choices=["ge", "le"],
        help="How to apply --conf_thresh: keep conf >= thresh ('ge', higher is more confident) or conf <= thresh ('le').",
    )
    ap.add_argument("--pred_use_lidar_mask", action="store_true", help="Filter predicted points by lidar valid_mask (often cleaner for OPV2V).")
    ap.add_argument("--depth_min", type=float, default=None, help="Optional depth(z) min filter in camera1 coords (None disables).")
    ap.add_argument("--depth_max", type=float, default=None, help="Optional depth(z) max filter in camera1 coords (None disables).")
    ap.add_argument("--completion_ratio_th", type=float, default=0.2)
    ap.add_argument("--sky_top_ratio", type=float, default=0.0, help="Drop top fraction of pixels (approx sky). 0 disables.")
    ap.add_argument("--stat_outlier_nb", type=int, default=0, help="Apply statistical outlier removal to pred PCD (0 disables).")
    ap.add_argument("--stat_outlier_std", type=float, default=2.0, help="Std ratio for statistical outlier removal.")
    ap.add_argument("--radius_outlier_nb", type=int, default=0, help="Apply radius outlier removal to pred PCD (0 disables).")
    ap.add_argument("--radius_outlier_radius", type=float, default=0.0, help="Radius for radius outlier removal.")
    ap.add_argument(
        "--semantic_drop_labels",
        type=str,
        default="",
        help="Comma-separated ADE20K labels to drop via SegFormer (e.g., 'sky' or 'sky,person,car'). Empty disables.",
    )
    ap.add_argument(
        "--semantic_model",
        type=str,
        default="nvidia/segformer-b0-finetuned-ade-512-512",
        help="HuggingFace model name for semantic segmentation.",
    )
    ap.add_argument(
        "--semantic_cache_dir",
        type=Path,
        default=_ROOT.parent / "pc_cache" / "segformer_ade20k",
        help="Cache dir for semantic label maps (speeds up repeated evals).",
    )
    ap.add_argument("--semantic_fp16", action="store_true", help="Use FP16 for semantic model on CUDA.")
    ap.add_argument("--save_vis_n", type=int, default=2, help="Save PLY overlays for first N scenes.")
    ap.add_argument("--out_dir", type=Path, required=True)
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    model = Spann3R(
        dus3r_name="./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
        use_feat=False,
    )
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model = model.to(args.device)
    model.eval()

    criterion = Regr3D_t_ScaleShiftInv(L21, norm_mode=False, gt_scale=True)

    cam_arg = str(args.camera).strip().lower()
    if cam_arg in ("all", "*"):
        cameras = ["camera0", "camera1", "camera2", "camera3"]
    else:
        cameras = [c.strip() for c in str(args.camera).split(",") if c.strip()]
    if not cameras:
        raise ValueError("No cameras selected. Use --camera camera0 or --camera camera0,camera1 or --camera all.")

    drop_labels = {s.strip().lower() for s in str(args.semantic_drop_labels).split(",") if s.strip()}
    segmenter = None
    drop_ids: list[int] = []
    if drop_labels:
        from driv3r.tools.semantic_mask import SegformerRunner

        if args.semantic_cache_dir is not None:
            args.semantic_cache_dir.mkdir(parents=True, exist_ok=True)
        segmenter = SegformerRunner.from_pretrained(
            model_name=str(args.semantic_model),
            device=str(args.device),
            fp16=bool(args.semantic_fp16),
        )
        drop_ids = segmenter.label_ids(drop_labels)
        if not drop_ids:
            print(f"[WARN] No valid labels matched in SegFormer config: {sorted(drop_labels)}. Disabling semantic mask.")
            segmenter = None

    root_dir = Path(args.data_root) / args.split
    scenario_folders = sorted([p for p in root_dir.iterdir() if p.is_dir()])

    results = []
    n_done = 0

    for scenario_folder in scenario_folders:
        if n_done >= int(args.max_scenes):
            break

        cav_list = [p.name for p in scenario_folder.iterdir() if p.is_dir()]
        if not cav_list:
            continue
        cav_list = _select_cavs(cav_list, args.max_cav if args.mode == "coop" else 1)

        # timestamps intersection across cavs
        ts_sets = []
        for cav_id in cav_list:
            cav_path = scenario_folder / cav_id
            yaml_files = sorted([p for p in cav_path.iterdir() if p.suffix == ".yaml" and "additional" not in p.name])
            timestamps = [p.stem for p in yaml_files]
            if len(timestamps) < int(args.sequence_length):
                ts_sets = []
                break
            ts_sets.append(set(timestamps))
        if not ts_sets:
            continue

        common = sorted(set.intersection(*ts_sets))
        if len(common) < int(args.sequence_length):
            continue

        timestamps = common[: int(args.sequence_length)]

        pred_pts_all = []
        pred_colors_all = []
        gt_pts_all = []

        # process each cav independently and union in world coords
        for cav_id in cav_list:
            cav_path = scenario_folder / cav_id

            for cam in cameras:
                split_arg = str(args.split_view).lower()
                splits = ["left", "right"] if split_arg in ("both", "*", "lr") else [split_arg]

                for split in splits:
                    frames = _build_views_for_sequence(
                        cav_path=cav_path,
                        timestamps=timestamps,
                        cam=cam,
                        resolution=args.resolution,
                        split=split,
                    )

                    # inference
                    for v in frames:
                        v["img"] = v["img"].to(args.device, non_blocking=True)
                    preds, preds_all = model.forward(frames, mem_sim_thresh=args.mem_sim_thresh)

                    # overwrite GT with lidar-projected sparse pointmaps
                    # (uses view["img"].device to allocate tensors)
                    from driv3r.datasets.opv2v import OPV2VDataset

                    frames = OPV2VDataset.load_lidar_pts(frames, image_size=(args.resolution, args.resolution))
                    for v in frames:
                        for k, val in v.items():
                            if isinstance(val, torch.Tensor):
                                v[k] = val.to(args.device, non_blocking=True)

                    gt_pts, pred_pts, _, _, _, monitoring = criterion.get_all_pts3d_t(frames, preds_all)
                    gt_shift_z = float(_to_numpy(monitoring["gt_shift_z"]).reshape(-1)[0])

                    cam1_c2w = frames[0]["camera_pose"][0].detach().cpu()

                    for j, view in enumerate(frames):
                        img = _to_numpy(view["img"].permute(0, 2, 3, 1)[0])
                        img01 = np.clip((img + 1.0) / 2.0, 0.0, 1.0)
                        if img01.shape[-1] != 3:
                            img01 = img01[..., :3]

                        lidar_mask = _to_numpy(view["valid_mask"][0]).astype(bool)
                        conf = _to_numpy(preds[j]["conf"][0]) if j < len(preds) else None

                        if j < len(pred_pts[0]):
                            pts_cam1 = _to_numpy(pred_pts[0][j][0])
                        else:
                            pts_cam1 = _to_numpy(pred_pts[1][-1][0])
                        pts_gt_cam1 = _to_numpy(gt_pts[j][0])

                        pts_cam1[..., 2] += gt_shift_z
                        pts_gt_cam1[..., 2] += gt_shift_z

                        drop_mask = None
                        if segmenter is not None and drop_ids:
                            img_u8 = (np.clip(img01, 0.0, 1.0) * 255.0).astype(np.uint8)
                            img_pil = Image.fromarray(img_u8, mode="RGB")
                            cache_key = (
                                f"{view.get('label','')}/{view.get('instance','')}/{split}/{args.resolution}"
                            )
                            label_map = segmenter.label_map(
                                img_pil,
                                cache_path=segmenter.cache_path(args.semantic_cache_dir, cache_key)
                                if args.semantic_cache_dir
                                else None,
                            )
                            drop_mask = np.isin(label_map, np.asarray(drop_ids, dtype=np.int64))

                        keep_pred = _mask_pred(
                            pts_cam1,
                            conf=conf,
                            conf_thresh=args.conf_thresh,
                            conf_keep=str(args.conf_keep),
                            conf_quantile=args.conf_quantile,
                            depth_min=args.depth_min,
                            depth_max=args.depth_max,
                            lidar_mask=lidar_mask if args.pred_use_lidar_mask else None,
                            drop_mask=drop_mask,
                            sky_top_ratio=args.sky_top_ratio,
                        )

                        pts_world = _to_numpy(geotrf(cam1_c2w, pts_cam1))
                        pts_gt_world = _to_numpy(geotrf(cam1_c2w, pts_gt_cam1))

                        finite_pred_w = np.isfinite(pts_world).all(axis=-1)
                        keep_pred &= finite_pred_w
                        pred_pts_all.append(pts_world[keep_pred].reshape(-1, 3))
                        pred_colors_all.append(img01[keep_pred].reshape(-1, 3))

                        finite_gt_w = np.isfinite(pts_gt_world).all(axis=-1)
                        keep_gt = finite_gt_w & lidar_mask
                        if drop_mask is not None:
                            # When semantic filtering is enabled, apply it to GT too (similar to
                            # nuScenes dynamic masking: exclude these regions from evaluation).
                            keep_gt &= ~drop_mask.astype(bool)
                        gt_pts_all.append(pts_gt_world[keep_gt].reshape(-1, 3))

        if not pred_pts_all or not gt_pts_all:
            continue

        pred_xyz = np.concatenate(pred_pts_all, axis=0)
        pred_rgb = np.concatenate(pred_colors_all, axis=0)
        gt_xyz = np.concatenate(gt_pts_all, axis=0)

        if pred_xyz.shape[0] < 1000 or gt_xyz.shape[0] < 1000:
            continue

        # Full-res PLYs for visualization.
        pcd_pred_full = _pcd_from(pred_xyz, pred_rgb)
        pcd_gt_full = _pcd_from(gt_xyz)
        pcd_gt_full.paint_uniform_color([0.25, 0.25, 0.25])

        # Downsampled point clouds for ICP + metrics (faster and stabler normals).
        pcd_pred = pcd_pred_full
        pcd_gt = pcd_gt_full
        if args.voxel_size and float(args.voxel_size) > 0:
            pcd_pred = pcd_pred_full.voxel_down_sample(float(args.voxel_size))
            pcd_gt = pcd_gt_full.voxel_down_sample(float(args.voxel_size))

        pcd_pred = _apply_outlier_removal(
            pcd_pred,
            stat_nb=int(args.stat_outlier_nb),
            stat_std=float(args.stat_outlier_std),
            radius_nb=int(args.radius_outlier_nb),
            radius=float(args.radius_outlier_radius),
        )

        if len(pcd_pred.points) < 100 or len(pcd_gt.points) < 100:
            continue

        # Raw (no ICP) metrics in world coords.
        pcd_pred_raw = copy.deepcopy(pcd_pred)
        pcd_gt_raw = copy.deepcopy(pcd_gt)
        pcd_pred_raw.estimate_normals()
        pcd_gt_raw.estimate_normals()
        gt_normal_raw = np.asarray(pcd_gt_raw.normals)
        pred_normal_raw = np.asarray(pcd_pred_raw.normals)
        acc_raw, acc_med_raw, nc1_raw, nc1_med_raw = accuracy(
            pcd_gt_raw.points, pcd_pred_raw.points, gt_normal_raw, pred_normal_raw
        )
        comp_raw, comp_med_raw, nc2_raw, nc2_med_raw = completion(
            pcd_gt_raw.points, pcd_pred_raw.points, gt_normal_raw, pred_normal_raw
        )
        cr_raw = completion_ratio(
            np.asarray(pcd_gt_raw.points),
            np.asarray(pcd_pred_raw.points),
            dist_th=float(args.completion_ratio_th),
        )

        trans_init = np.eye(4, dtype=np.float64)
        reg = o3d.pipelines.registration.registration_icp(
            pcd_pred,
            pcd_gt,
            float(args.icp_threshold),
            trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )
        T = reg.transformation

        pcd_pred_icp = copy.deepcopy(pcd_pred)
        pcd_pred_icp.transform(T)
        pcd_pred_icp.estimate_normals()
        pcd_gt.estimate_normals()

        gt_normal = np.asarray(pcd_gt.normals)
        pred_normal = np.asarray(pcd_pred_icp.normals)

        acc, acc_med, nc1, nc1_med = accuracy(pcd_gt.points, pcd_pred_icp.points, gt_normal, pred_normal)
        comp, comp_med, nc2, nc2_med = completion(pcd_gt.points, pcd_pred_icp.points, gt_normal, pred_normal)
        cr = completion_ratio(
            np.asarray(pcd_gt.points),
            np.asarray(pcd_pred_icp.points),
            dist_th=float(args.completion_ratio_th),
        )

        pred_center = np.asarray(pcd_pred.get_center(), dtype=np.float64)
        gt_center = np.asarray(pcd_gt.get_center(), dtype=np.float64)
        rot_deg = _rot_angle_deg(np.asarray(T, dtype=np.float64))
        icp_t_norm = float(np.linalg.norm(np.asarray(T, dtype=np.float64)[:3, 3]))
        icp_centered_t_norm = _centered_icp_t_norm(np.asarray(T, dtype=np.float64), pred_center, gt_center)
        center_diff_norm = float(np.linalg.norm(gt_center - pred_center))

        scene_res = dict(
            scenario=scenario_folder.name,
            cav_ids=cav_list,
            cameras=cameras,
            split_view=args.split_view,
            n_pred=int(len(pcd_pred.points)),
            n_gt=int(len(pcd_gt.points)),
            n_pred_full=int(len(pcd_pred_full.points)),
            n_gt_full=int(len(pcd_gt_full.points)),
            icp_fitness=float(reg.fitness),
            icp_rmse=float(reg.inlier_rmse),
            icp_rot_deg=float(rot_deg),
            icp_t_norm=float(icp_t_norm),
            icp_centered_t_norm=float(icp_centered_t_norm),
            centroid_diff_norm=float(center_diff_norm),
            accuracy_raw=float(acc_raw),
            completion_raw=float(comp_raw),
            nc1_raw=float(nc1_raw),
            nc2_raw=float(nc2_raw),
            acc_med_raw=float(acc_med_raw),
            comp_med_raw=float(comp_med_raw),
            nc1_med_raw=float(nc1_med_raw),
            nc2_med_raw=float(nc2_med_raw),
            completion_ratio_raw=float(cr_raw),
            accuracy=float(acc),
            completion=float(comp),
            nc1=float(nc1),
            nc2=float(nc2),
            acc_med=float(acc_med),
            comp_med=float(comp_med),
            nc1_med=float(nc1_med),
            nc2_med=float(nc2_med),
            completion_ratio=float(cr),
        )
        results.append(scene_res)

        if n_done < int(args.save_vis_n):
            scene_dir = out_dir / f"scene_{n_done:04d}_{scenario_folder.name}"
            scene_dir.mkdir(parents=True, exist_ok=True)

            pcd_pred_save = pcd_pred_full
            pcd_gt_save = pcd_gt_full
            if args.vis_voxel_size and float(args.vis_voxel_size) > 0:
                pcd_pred_save = pcd_pred_full.voxel_down_sample(float(args.vis_voxel_size))
                pcd_gt_save = pcd_gt_full.voxel_down_sample(float(args.vis_voxel_size))

            pcd_pred_save = _apply_outlier_removal(
                pcd_pred_save,
                stat_nb=int(args.stat_outlier_nb),
                stat_std=float(args.stat_outlier_std),
                radius_nb=int(args.radius_outlier_nb),
                radius=float(args.radius_outlier_radius),
            )

            pcd_pred_icp_save = copy.deepcopy(pcd_pred_save)
            pcd_pred_icp_save.transform(T)

            o3d.io.write_point_cloud(str(scene_dir / "pred_raw.ply"), pcd_pred_save)
            o3d.io.write_point_cloud(str(scene_dir / "pred_icp.ply"), pcd_pred_icp_save)
            o3d.io.write_point_cloud(str(scene_dir / "gt.ply"), pcd_gt_save)
            overlay = copy.deepcopy(pcd_gt_save)
            overlay += pcd_pred_icp_save
            o3d.io.write_point_cloud(str(scene_dir / "overlay_gt_gray_pred_color.ply"), overlay)
            (scene_dir / "metrics.json").write_text(json.dumps(scene_res, indent=2))

        print(
            f"[{n_done:04d}] {scenario_folder.name} cavs={len(cav_list)} "
            f"Acc(raw)={scene_res['accuracy_raw']:.3f} Acc(icp)={scene_res['accuracy']:.3f} "
            f"Comp(raw)={scene_res['completion_raw']:.3f} Comp(icp)={scene_res['completion']:.3f} "
            f"CR@{args.completion_ratio_th:.2f} raw={scene_res['completion_ratio_raw']:.3f} icp={scene_res['completion_ratio']:.3f} "
            f"(pred={scene_res['n_pred']} gt={scene_res['n_gt']})"
        )

        n_done += 1

    # summarize
    def _avg(key: str):
        vals = [r[key] for r in results if key in r]
        return float(np.mean(vals)) if vals else float("nan")

    summary = dict(
        n_scenes=int(len(results)),
        mode=args.mode,
        max_cav=int(args.max_cav),
        cameras=cameras,
        split_view=args.split_view,
        resolution=int(args.resolution),
        sequence_length=int(args.sequence_length),
        voxel_size=float(args.voxel_size),
        vis_voxel_size=float(args.vis_voxel_size),
        conf_thresh=None if args.conf_thresh is None else float(args.conf_thresh),
        conf_keep=str(args.conf_keep),
        pred_use_lidar_mask=bool(args.pred_use_lidar_mask),
        depth_min=args.depth_min,
        depth_max=args.depth_max,
        icp_threshold=float(args.icp_threshold),
        completion_ratio_th=float(args.completion_ratio_th),
        sky_top_ratio=float(args.sky_top_ratio),
        semantic_drop_labels=sorted(drop_labels),
        semantic_model=str(args.semantic_model),
        semantic_cache_dir=str(args.semantic_cache_dir) if args.semantic_cache_dir is not None else None,
        semantic_fp16=bool(args.semantic_fp16),
        icp_rot_deg=_avg("icp_rot_deg"),
        icp_t_norm=_avg("icp_t_norm"),
        icp_centered_t_norm=_avg("icp_centered_t_norm"),
        centroid_diff_norm=_avg("centroid_diff_norm"),
        accuracy_raw=_avg("accuracy_raw"),
        completion_raw=_avg("completion_raw"),
        nc1_raw=_avg("nc1_raw"),
        nc2_raw=_avg("nc2_raw"),
        acc_med_raw=_avg("acc_med_raw"),
        comp_med_raw=_avg("comp_med_raw"),
        nc1_med_raw=_avg("nc1_med_raw"),
        nc2_med_raw=_avg("nc2_med_raw"),
        completion_ratio_raw=_avg("completion_ratio_raw"),
        accuracy=_avg("accuracy"),
        completion=_avg("completion"),
        nc1=_avg("nc1"),
        nc2=_avg("nc2"),
        acc_med=_avg("acc_med"),
        comp_med=_avg("comp_med"),
        nc1_med=_avg("nc1_med"),
        nc2_med=_avg("nc2_med"),
        completion_ratio=_avg("completion_ratio"),
    )

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "per_scene.json").write_text(json.dumps(results, indent=2))
    print("[DONE] summary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
