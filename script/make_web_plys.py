import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


def _sample_points(pcd: o3d.geometry.PointCloud, max_points: int, seed: int) -> o3d.geometry.PointCloud:
    n = int(np.asarray(pcd.points).shape[0])
    if max_points <= 0 or n <= max_points:
        return pcd
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(n, size=int(max_points), replace=False)
    return pcd.select_by_index(idx.tolist())


def _make_out_path(ply_path: Path, suffix: str) -> Path:
    stem = ply_path.stem
    if stem.endswith(suffix):
        return ply_path
    return ply_path.with_name(f"{stem}{suffix}{ply_path.suffix}")


def main():
    ap = argparse.ArgumentParser("Downsample large PLYs for faster web viewing (writes *_web.ply next to originals).")
    ap.add_argument("--root", type=Path, required=True, help="Root directory to scan (recursively).")
    ap.add_argument(
        "--names",
        type=str,
        default="gt.ply,pred_icp.ply",
        help="Comma-separated PLY basenames to process.",
    )
    ap.add_argument("--suffix", type=str, default="_web", help="Suffix for output PLYs (default: _web).")
    ap.add_argument("--voxel_size", type=float, default=0.15, help="Voxel downsample size (0 disables).")
    ap.add_argument("--max_points", type=int, default=500000, help="Randomly subsample to at most this many points (0 disables).")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for point sampling.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing *_web.ply files.")
    args = ap.parse_args()

    root = args.root
    names = {n.strip() for n in str(args.names).split(",") if n.strip()}
    if not names:
        raise ValueError("No --names specified.")

    voxel_size = float(args.voxel_size)
    suffix = str(args.suffix)

    total_in = 0
    total_out = 0

    for name in sorted(names):
        for ply_path in sorted(root.rglob(name)):
            if not ply_path.is_file():
                continue
            out_path = _make_out_path(ply_path, suffix)
            if out_path.exists() and not args.overwrite:
                continue

            pcd = o3d.io.read_point_cloud(str(ply_path))
            if voxel_size and voxel_size > 0:
                pcd = pcd.voxel_down_sample(voxel_size)
            pcd = _sample_points(pcd, int(args.max_points), int(args.seed))

            o3d.io.write_point_cloud(str(out_path), pcd, write_ascii=False, compressed=False, print_progress=False)
            total_in += 1
            total_out += 1
            print(f"[OK] {ply_path} -> {out_path} (n={len(pcd.points)})")

    print(f"[DONE] processed={total_in} wrote={total_out} under: {root}")


if __name__ == "__main__":
    main()

