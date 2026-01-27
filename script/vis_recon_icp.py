import argparse
import copy
import json
import os
from pathlib import Path
import sys

# Allow running as `python3 script/vis_recon_icp.py` from anywhere.
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import numpy as np
import open3d as o3d
import torch

from dust3r.losses import L21
from dust3r.utils.geometry import geotrf
from driv3r.datasets import NuSceneDataset, build_dataset
from driv3r.loss import Regr3D_t_ScaleShiftInv
from driv3r.model import Spann3R
from driv3r.tools.eval_recon import accuracy, completion, completion_ratio


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _safe_colors(img01):
    img01 = np.clip(img01, 0.0, 1.0)
    if img01.ndim == 2:
        return np.repeat(img01[..., None], 3, axis=-1)
    if img01.shape[-1] == 1:
        return np.repeat(img01, 3, axis=-1)
    return img01[..., :3]


def _pcd_from(points_xyz, colors_rgb=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz.astype(np.float64))
    if colors_rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors_rgb.astype(np.float64))
    return pcd


def _rot_angle_deg(R: np.ndarray) -> float:
    tr = float(np.trace(R[:3, :3]))
    c = (tr - 1.0) / 2.0
    c = float(np.clip(c, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def _centered_icp_t_norm(T: np.ndarray, pred_center: np.ndarray, gt_center: np.ndarray) -> float:
    A = np.eye(4, dtype=np.float64)
    A[:3, 3] = pred_center.reshape(3)
    B = np.eye(4, dtype=np.float64)
    B[:3, 3] = -gt_center.reshape(3)
    Tc = B @ T @ A
    return float(np.linalg.norm(Tc[:3, 3]))


def _filter_mask(
    *,
    finite_mask,
    lidar_mask=None,
    conf=None,
    conf_thresh=None,
    depth=None,
    depth_min=None,
    depth_max=None,
    drop_mask=None,
    sky_top_ratio=None,
    height=None,
    width=None,
    mask_mode="lidar+conf+depth",
):
    keep = finite_mask.copy()

    if depth is not None and (depth_min is not None or depth_max is not None):
        if depth_min is None:
            depth_min = -np.inf
        if depth_max is None:
            depth_max = np.inf
        keep &= (depth >= float(depth_min)) & (depth <= float(depth_max))

    if lidar_mask is not None and ("lidar" in mask_mode):
        keep &= lidar_mask.astype(bool)

    if conf is not None and conf_thresh is not None and ("conf" in mask_mode):
        # In DUSt3R/Driv3R, higher conf generally correlates with lower 3D regression error.
        keep &= conf >= float(conf_thresh)

    if drop_mask is not None:
        keep &= ~drop_mask.astype(bool)

    if sky_top_ratio is not None and float(sky_top_ratio) > 0 and height is not None and width is not None:
        h = int(height)
        w = int(width)
        sky_h = int(round(h * float(sky_top_ratio)))
        if sky_h > 0:
            sky = np.zeros((h, w), dtype=bool)
            sky[:sky_h, :] = True
            keep &= ~sky

    return keep


@torch.no_grad()
def main():
    p = argparse.ArgumentParser("Create ICP-aligned PLYs + recon metrics for a few sequences.")
    p.add_argument("--ckpt_path", type=str, required=True)
    p.add_argument("--dataset", type=str, required=True, help="Dataset string (same as eval.py).")
    p.add_argument("--resolution", type=int, required=True)
    p.add_argument("--sequence_length", type=int, required=True)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--max_batches", type=int, default=8)
    p.add_argument("--mask_mode", type=str, default="lidar+conf+depth", choices=["all", "lidar", "conf", "depth", "lidar+conf", "lidar+depth", "conf+depth", "lidar+conf+depth"])
    p.add_argument(
        "--conf_thresh",
        type=float,
        default=1.0,
        help="Keep pixels with conf >= thresh (higher is higher confidence). Set 1.0 to effectively disable (conf>=1).",
    )
    p.add_argument("--depth_min", type=float, default=0.5)
    p.add_argument("--depth_max", type=float, default=80.0)
    p.add_argument("--splits", type=str, default=None, help="Comma-separated subset of splits to run (e.g., 'left' or 'left,right' or 'single'). Defaults to all available.")
    p.add_argument("--icp_threshold", type=float, default=100.0)
    p.add_argument("--completion_ratio_th", type=float, default=0.2)
    p.add_argument("--sky_top_ratio", type=float, default=0.0, help="Drop top fraction of pixels (approx sky). 0 disables.")
    p.add_argument(
        "--semantic_drop_labels",
        type=str,
        default="",
        help="Comma-separated ADE20K labels to drop via SegFormer (e.g., 'sky,person,car'). Empty disables.",
    )
    p.add_argument(
        "--semantic_model",
        type=str,
        default="nvidia/segformer-b0-finetuned-ade-512-512",
        help="HuggingFace model name for semantic segmentation.",
    )
    p.add_argument(
        "--semantic_cache_dir",
        type=Path,
        default=_ROOT.parent / "pc_cache" / "segformer_ade20k",
        help="Cache dir for semantic label maps (speeds up repeated evals).",
    )
    p.add_argument("--semantic_fp16", action="store_true", help="Use FP16 for semantic model on CUDA.")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = Spann3R(
        dus3r_name="./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
        use_feat=False,
    )
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model = model.to(args.device)
    model.eval()

    criterion = Regr3D_t_ScaleShiftInv(L21, norm_mode=False, gt_scale=True)
    dataloader = build_dataset(args.dataset, batch_size=1, num_workers=0, test=True)
    dataset = dataloader.dataset

    summary = []
    threshold = float(args.icp_threshold)

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

    for batch_id, batch in enumerate(dataloader):
        if args.max_batches and batch_id >= int(args.max_batches):
            break

        for view in batch:
            view["img"] = view["img"].to(args.device, non_blocking=True)

        # Support both (sequence_length) and (2*sequence_length) batches.
        if len(batch) == 2 * args.sequence_length:
            splits = {
                "left": batch[0::2],
                "right": batch[1::2],
            }
        elif len(batch) == args.sequence_length:
            splits = {"single": batch}
        else:
            raise ValueError(f"Unexpected batch length: {len(batch)}")

        if args.splits:
            allowed = {s.strip() for s in str(args.splits).split(",") if s.strip()}
            splits = {k: v for k, v in splits.items() if k in allowed}
            if not splits:
                continue

        for split_name, split_batch in splits.items():
            preds, preds_all = model.forward(split_batch)

            # overwrite GT with lidar-projected points (sparse but accurate)
            if hasattr(dataset, "load_lidar_pts"):
                split_batch = dataset.load_lidar_pts(split_batch, image_size=(args.resolution, args.resolution))
            else:
                split_batch = NuSceneDataset.load_lidar_pts(split_batch, image_size=(args.resolution, args.resolution))
            for view in split_batch:
                for k, v in view.items():
                    if isinstance(v, torch.Tensor):
                        view[k] = v.to(args.device, non_blocking=True)

            gt_pts, pred_pts, _, _, _, monitoring = criterion.get_all_pts3d_t(split_batch, preds_all)
            gt_shift_z = monitoring["gt_shift_z"]

            # camera1 pose (cam2world) for mapping camera1-coords -> world
            cam1_c2w = split_batch[0]["camera_pose"][0].detach().cpu()

            pred_points_all = []
            pred_colors_all = []
            gt_points_all = []

            for j, view in enumerate(split_batch):
                img = _to_numpy(view["img"].permute(0, 2, 3, 1)[0])
                img01 = (img + 1.0) / 2.0
                img01 = _safe_colors(img01)

                lidar_mask = _to_numpy(view["valid_mask"][0]).astype(bool)

                conf = _to_numpy(preds[j]["conf"][0]) if j < len(preds) else None

                # pick a prediction map for every view index (same logic as eval.py)
                if j < len(pred_pts[0]):
                    pts_cam1 = _to_numpy(pred_pts[0][j][0])
                else:
                    pts_cam1 = _to_numpy(pred_pts[1][-1][0])

                pts_gt_cam1 = _to_numpy(gt_pts[j][0])

                # restore shift and compute depth in camera1 coordinates
                shift = float(_to_numpy(gt_shift_z).reshape(-1)[0])
                pts_cam1[..., 2] += shift
                pts_gt_cam1[..., 2] += shift
                depth = pts_cam1[..., 2]

                # transform to world
                pts_world = _to_numpy(geotrf(cam1_c2w, pts_cam1))
                pts_gt_world = _to_numpy(geotrf(cam1_c2w, pts_gt_cam1))

                drop_mask = None
                if segmenter is not None and drop_ids:
                    img_u8 = (np.clip(img01, 0.0, 1.0) * 255.0).astype(np.uint8)
                    from PIL import Image

                    img_pil = Image.fromarray(img_u8, mode="RGB")
                    cache_key = f"{view.get('label','')}/{view.get('instance','')}/{split_name}/{args.resolution}"
                    label_map = segmenter.label_map(
                        img_pil,
                        cache_path=segmenter.cache_path(args.semantic_cache_dir, cache_key) if args.semantic_cache_dir else None,
                    )
                    drop_mask = np.isin(label_map, np.asarray(drop_ids, dtype=np.int64))

                finite_pred = np.isfinite(pts_world).all(axis=-1)
                keep_pred = _filter_mask(
                    finite_mask=finite_pred,
                    lidar_mask=lidar_mask,
                    conf=conf,
                    conf_thresh=args.conf_thresh,
                    depth=depth,
                    depth_min=args.depth_min,
                    depth_max=args.depth_max,
                    drop_mask=drop_mask,
                    sky_top_ratio=args.sky_top_ratio,
                    height=pts_world.shape[0],
                    width=pts_world.shape[1],
                    mask_mode=args.mask_mode,
                )

                pred_points_all.append(pts_world[keep_pred].reshape(-1, 3))
                pred_colors_all.append(img01[keep_pred].reshape(-1, 3))

                finite_gt = np.isfinite(pts_gt_world).all(axis=-1)
                keep_gt = finite_gt & lidar_mask
                gt_points_all.append(pts_gt_world[keep_gt].reshape(-1, 3))

            pred_points = np.concatenate(pred_points_all, axis=0) if pred_points_all else np.zeros((0, 3), dtype=np.float32)
            pred_colors = np.concatenate(pred_colors_all, axis=0) if pred_colors_all else np.zeros((0, 3), dtype=np.float32)
            gt_points = np.concatenate(gt_points_all, axis=0) if gt_points_all else np.zeros((0, 3), dtype=np.float32)

            # guard empty point clouds
            if pred_points.shape[0] < 50 or gt_points.shape[0] < 50:
                continue

            pcd_pred = _pcd_from(pred_points, pred_colors)
            pcd_gt = _pcd_from(gt_points)
            pcd_gt.paint_uniform_color([0.25, 0.25, 0.25])

            # Raw (no ICP) metrics.
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
            comp_ratio_raw = completion_ratio(
                np.asarray(pcd_gt_raw.points),
                np.asarray(pcd_pred_raw.points),
                dist_th=float(args.completion_ratio_th),
            )

            trans_init = np.eye(4, dtype=np.float64)
            reg_p2p = o3d.pipelines.registration.registration_icp(
                pcd_pred,
                pcd_gt,
                threshold,
                trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            )
            T = reg_p2p.transformation

            pcd_pred_aligned = copy.deepcopy(pcd_pred)
            pcd_pred_aligned.transform(T)
            pcd_pred_aligned.estimate_normals()
            pcd_gt.estimate_normals()

            gt_normal = np.asarray(pcd_gt.normals)
            pred_normal = np.asarray(pcd_pred_aligned.normals)

            acc, acc_med, nc1, nc1_med = accuracy(pcd_gt.points, pcd_pred_aligned.points, gt_normal, pred_normal)
            comp, comp_med, nc2, nc2_med = completion(pcd_gt.points, pcd_pred_aligned.points, gt_normal, pred_normal)
            comp_ratio = completion_ratio(np.asarray(pcd_gt.points), np.asarray(pcd_pred_aligned.points), dist_th=float(args.completion_ratio_th))

            pred_center = np.asarray(pcd_pred.get_center(), dtype=np.float64)
            gt_center = np.asarray(pcd_gt.get_center(), dtype=np.float64)
            rot_deg = _rot_angle_deg(np.asarray(T, dtype=np.float64))
            icp_t_norm = float(np.linalg.norm(np.asarray(T, dtype=np.float64)[:3, 3]))
            icp_centered_t_norm = _centered_icp_t_norm(np.asarray(T, dtype=np.float64), pred_center, gt_center)
            center_diff_norm = float(np.linalg.norm(gt_center - pred_center))

            run_dir = args.out_dir / f"batch{batch_id:06d}_{split_name}"
            run_dir.mkdir(parents=True, exist_ok=True)
            o3d.io.write_point_cloud(str(run_dir / "pred_raw.ply"), pcd_pred)
            o3d.io.write_point_cloud(str(run_dir / "pred_icp.ply"), pcd_pred_aligned)
            o3d.io.write_point_cloud(str(run_dir / "gt.ply"), pcd_gt)

            overlay = copy.deepcopy(pcd_gt)
            overlay += pcd_pred_aligned
            o3d.io.write_point_cloud(str(run_dir / "overlay_gt_gray_pred_color.ply"), overlay)

            meta = {
                "batch_id": int(batch_id),
                "split": split_name,
                "n_pred": int(np.asarray(pcd_pred.points).shape[0]),
                "n_gt": int(np.asarray(pcd_gt.points).shape[0]),
                "icp_fitness": float(reg_p2p.fitness),
                "icp_rmse": float(reg_p2p.inlier_rmse),
                "icp_T": np.asarray(T, dtype=np.float64).tolist(),
                "icp_rot_deg": float(rot_deg),
                "icp_t_norm": float(icp_t_norm),
                "icp_centered_t_norm": float(icp_centered_t_norm),
                "centroid_diff_norm": float(center_diff_norm),
                "accuracy_raw": float(acc_raw),
                "completion_raw": float(comp_raw),
                "nc1_raw": float(nc1_raw),
                "nc2_raw": float(nc2_raw),
                "acc_med_raw": float(acc_med_raw),
                "comp_med_raw": float(comp_med_raw),
                "nc1_med_raw": float(nc1_med_raw),
                "nc2_med_raw": float(nc2_med_raw),
                "completion_ratio_raw": float(comp_ratio_raw),
                "accuracy": float(acc),
                "completion": float(comp),
                "nc1": float(nc1),
                "nc2": float(nc2),
                "acc_med": float(acc_med),
                "comp_med": float(comp_med),
                "nc1_med": float(nc1_med),
                "nc2_med": float(nc2_med),
                "completion_ratio": float(comp_ratio),
            }
            (run_dir / "metrics.json").write_text(json.dumps(meta, indent=2))
            summary.append(meta)

            print(
                f"[{batch_id:04d}/{split_name}] "
                f"Acc(raw)={acc_raw:.3f} Acc(icp)={acc:.3f} "
                f"Comp(raw)={comp_raw:.3f} Comp(icp)={comp:.3f} "
                f"CR@{args.completion_ratio_th:.2f} raw={comp_ratio_raw:.3f} icp={comp_ratio:.3f} "
                f"(pred={meta['n_pred']} gt={meta['n_gt']})"
            )

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[OK] Wrote: {args.out_dir}/summary.json")


if __name__ == "__main__":
    main()
