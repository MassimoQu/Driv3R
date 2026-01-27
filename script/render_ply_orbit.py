import os

# Must be set before importing Open3D for headless rendering.
os.environ.setdefault("OPEN3D_CPU_RENDERING", "true")

import argparse
import math
from pathlib import Path

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


def _camera_orbit(center: np.ndarray, radius: float, elevation: float, theta: float) -> np.ndarray:
    return center + np.array(
        [radius * math.cos(theta), radius * math.sin(theta), radius * elevation],
        dtype=np.float32,
    )


def _render_rgb(renderer: o3d.visualization.rendering.OffscreenRenderer) -> np.ndarray:
    img = renderer.render_to_image()
    rgb = np.asarray(img)
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8)
    return rgb


def _write_video(frames_rgb, out_mp4: Path, fps: int):
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    frames_rgb = iter(frames_rgb)
    first = next(frames_rgb, None)
    if first is None:
        raise ValueError("No frames to write.")

    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_mp4), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("Failed to open cv2.VideoWriter (mp4v).")

    def _to_bgr(img_rgb: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    writer.write(_to_bgr(first))
    for rgb in frames_rgb:
        writer.write(_to_bgr(rgb))
    writer.release()


def main():
    p = argparse.ArgumentParser("Render an orbit video (and snapshots) for a point cloud PLY.")
    p.add_argument("--input_ply", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--name", type=str, default=None, help="Output filename prefix (defaults to PLY stem).")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--n_frames", type=int, default=180)
    p.add_argument("--fov", type=float, default=60.0)
    p.add_argument("--radius_scale", type=float, default=1.5)
    p.add_argument("--elevation", type=float, default=0.25, help="Relative elevation (z = radius * elevation).")
    p.add_argument("--point_size", type=float, default=2.0)
    p.add_argument("--voxel_size", type=float, default=0.15, help="0 disables downsampling.")
    p.add_argument("--background", choices=["white", "black"], default="white")
    p.add_argument("--save_snapshots", action="store_true", help="Also save 5 PNG snapshots.")
    args = p.parse_args()

    if not args.input_ply.exists():
        raise FileNotFoundError(args.input_ply)

    name = args.name or args.input_ply.stem
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = out_dir / f"{name}_orbit.mp4"

    pcd = o3d.io.read_point_cloud(str(args.input_ply))
    if args.voxel_size and args.voxel_size > 0:
        pcd = pcd.voxel_down_sample(float(args.voxel_size))
    if not pcd.has_colors():
        pcd.paint_uniform_color([0.6, 0.6, 0.6])

    renderer = _build_renderer(args.width, args.height, args.background)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = float(args.point_size)
    renderer.scene.add_geometry("pcd", pcd, mat)

    bbox = renderer.scene.bounding_box
    center = np.asarray(bbox.get_center(), dtype=np.float32)
    extent = np.asarray(bbox.get_extent(), dtype=np.float32)
    radius = float(np.linalg.norm(extent) * float(args.radius_scale))
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    def frames():
        for i in range(int(args.n_frames)):
            theta = (2.0 * math.pi * i) / float(args.n_frames)
            eye = _camera_orbit(center, radius, float(args.elevation), theta)
            renderer.setup_camera(float(args.fov), center, eye, up)
            yield _render_rgb(renderer)

    _write_video(frames(), out_mp4, int(args.fps))

    if args.save_snapshots:
        snapshot_angles = [0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi, 0.25 * math.pi]
        for idx, theta in enumerate(snapshot_angles):
            eye = _camera_orbit(center, radius, float(args.elevation), theta)
            renderer.setup_camera(float(args.fov), center, eye, up)
            rgb = _render_rgb(renderer)
            out_png = out_dir / f"{name}_view{idx}.png"
            cv2.imwrite(str(out_png), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    print(f"[OK] Wrote video: {out_mp4}")
    if args.save_snapshots:
        print(f"[OK] Wrote snapshots: {out_dir}/{name}_view*.png")


if __name__ == "__main__":
    main()

