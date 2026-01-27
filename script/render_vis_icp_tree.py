import argparse
import os
from pathlib import Path

# Must be set before importing Open3D for headless rendering.
os.environ.setdefault("OPEN3D_CPU_RENDERING", "true")

import cv2
import numpy as np
import open3d as o3d


def _build_renderer(width: int, height: int, background: str):
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    if background == "white":
        renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
    elif background == "black":
        renderer.scene.set_background([0.0, 0.0, 0.0, 1.0])
    else:
        raise ValueError(f"Unsupported background: {background}")
    return renderer


def _render_orbit(
    pcd: o3d.geometry.PointCloud,
    out_dir: Path,
    name: str,
    *,
    width: int,
    height: int,
    fps: int,
    n_frames: int,
    fov: float,
    radius_scale: float,
    elevation: float,
    point_size: float,
    voxel_size: float,
    background: str,
    save_snapshots: bool,
):
    if voxel_size and voxel_size > 0:
        pcd = pcd.voxel_down_sample(float(voxel_size))
    if not pcd.has_colors():
        pcd.paint_uniform_color([0.7, 0.7, 0.7])

    out_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = out_dir / f"{name}_orbit.mp4"

    renderer = _build_renderer(width, height, background)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = float(point_size)
    renderer.scene.add_geometry("pcd", pcd, mat)

    bbox = renderer.scene.bounding_box
    center = np.asarray(bbox.get_center(), dtype=np.float32)
    extent = np.asarray(bbox.get_extent(), dtype=np.float32)
    radius = float(np.linalg.norm(extent) * float(radius_scale))
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    def _render_rgb():
        img = renderer.render_to_image()
        rgb = np.asarray(img)
        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8)
        return rgb

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_mp4), fourcc, int(fps), (int(width), int(height)))
    if not writer.isOpened():
        raise RuntimeError("Failed to open cv2.VideoWriter (mp4v).")

    def _orbit_eye(theta: float):
        return center + np.array(
            [radius * float(np.cos(theta)), radius * float(np.sin(theta)), radius * float(elevation)],
            dtype=np.float32,
        )

    for i in range(int(n_frames)):
        theta = (2.0 * np.pi * i) / float(max(int(n_frames), 1))
        eye = _orbit_eye(theta)
        renderer.setup_camera(float(fov), center, eye, up)
        rgb = _render_rgb()
        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    writer.release()

    if save_snapshots:
        snapshot_angles = [0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi, 0.25 * np.pi]
        for idx, theta in enumerate(snapshot_angles):
            eye = _orbit_eye(theta)
            renderer.setup_camera(float(fov), center, eye, up)
            rgb = _render_rgb()
            cv2.imwrite(str(out_dir / f"{name}_view{idx}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def main():
    ap = argparse.ArgumentParser("Render all ICP visualization folders under a root directory.")
    ap.add_argument("--root", type=Path, required=True, help="Root directory produced by vis scripts.")
    ap.add_argument("--ply_name", type=str, default="overlay_gt_gray_pred_color.ply")
    ap.add_argument("--name", type=str, default="overlay", help="Output prefix (overlay -> overlay_orbit.mp4, overlay_view*.png)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--n_frames", type=int, default=180)
    ap.add_argument("--fov", type=float, default=60.0)
    ap.add_argument("--radius_scale", type=float, default=0.5)
    ap.add_argument("--elevation", type=float, default=0.25)
    ap.add_argument("--point_size", type=float, default=2.0)
    ap.add_argument("--voxel_size", type=float, default=0.2)
    ap.add_argument("--background", choices=["white", "black"], default="black")
    ap.add_argument("--save_snapshots", action="store_true")
    args = ap.parse_args()

    ply_files = sorted(args.root.rglob(args.ply_name))
    if not ply_files:
        raise FileNotFoundError(f"No '{args.ply_name}' found under: {args.root}")

    for ply in ply_files:
        out_dir = ply.parent
        pcd = o3d.io.read_point_cloud(str(ply))
        _render_orbit(
            pcd,
            out_dir,
            args.name,
            width=args.width,
            height=args.height,
            fps=args.fps,
            n_frames=args.n_frames,
            fov=args.fov,
            radius_scale=args.radius_scale,
            elevation=args.elevation,
            point_size=args.point_size,
            voxel_size=args.voxel_size,
            background=args.background,
            save_snapshots=args.save_snapshots,
        )
        print(f"[OK] Rendered: {ply} -> {out_dir}/{args.name}_orbit.mp4")


if __name__ == "__main__":
    main()

