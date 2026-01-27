import os
import json
import pickle
import argparse
import tqdm


def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _pick_version_dir(data_root: str) -> str:
    for candidate in ("v1.0-trainval", "v1.0-mini", "v1.0-test"):
        cand = os.path.join(data_root, candidate)
        if os.path.isdir(cand):
            return cand
    raise FileNotFoundError(
        f"Cannot find nuScenes metadata folder under {data_root}. Expected one of: v1.0-trainval, v1.0-mini, v1.0-test"
    )


def main():
    parser = argparse.ArgumentParser("Generate train_metas.pkl / val_metas.pkl for Driv3R nuScenes loader")
    parser.add_argument("--data_root", required=True, type=str, help="nuScenes root (contains samples/ and v1.0-*/)")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="scene-level val split ratio")
    args = parser.parse_args()

    data_root = args.data_root
    version_dir = _pick_version_dir(data_root)

    sensors = _load_json(os.path.join(version_dir, "sensor.json"))
    calibrated_sensors = _load_json(os.path.join(version_dir, "calibrated_sensor.json"))
    ego_poses = _load_json(os.path.join(version_dir, "ego_pose.json"))
    samples = _load_json(os.path.join(version_dir, "sample.json"))
    scenes = _load_json(os.path.join(version_dir, "scene.json"))
    sample_datas = _load_json(os.path.join(version_dir, "sample_data.json"))

    sensor_maps = {s["token"]: s["channel"] for s in sensors}

    calibrated_sensor_maps = {}
    for cs in calibrated_sensors:
        calibrated_sensor_maps[cs["token"]] = dict(
            channel=sensor_maps[cs["sensor_token"]],
            translation=cs["translation"],
            rotation=cs["rotation"],
            intrinsics=cs.get("camera_intrinsic", []),
        )

    ego_pose_maps = {ep["token"]: dict(translation=ep["translation"], rotation=ep["rotation"]) for ep in ego_poses}
    sample_to_scene = {s["token"]: s["scene_token"] for s in samples}

    scene_dict = {s["name"]: s["token"] for s in scenes}
    scene_names_sorted = sorted(scene_dict.keys())
    scene_tokens = [scene_dict[name] for name in scene_names_sorted]

    sample_data_maps = {scene_token: {} for scene_token in scene_tokens}

    for sd in tqdm.tqdm(sample_datas, desc="loading camera keyframes"):
        if sd.get("fileformat") != "jpg" or not sd.get("is_key_frame", False):
            continue
        sample_token = sd["sample_token"]
        scene_token = sample_to_scene[sample_token]
        if sample_token not in sample_data_maps[scene_token]:
            sample_data_maps[scene_token][sample_token] = {}

        calibrated_sensor = calibrated_sensor_maps[sd["calibrated_sensor_token"]]
        ego_pose = ego_pose_maps[sd["ego_pose_token"]]
        channel = calibrated_sensor["channel"]

        sample_data_maps[scene_token][sample_token][channel] = dict(
            timestamp=sd["timestamp"],
            filename=sd["filename"],
            ego_pose=ego_pose,
            calibrated_sensor=calibrated_sensor,
            image_wh=(sd.get("width", None), sd.get("height", None)),
        )

    for sd in tqdm.tqdm(sample_datas, desc="loading lidar keyframes"):
        if not sd.get("is_key_frame", False):
            continue
        filename = sd.get("filename", "")
        if "LIDAR_TOP" not in filename:
            continue
        sample_token = sd["sample_token"]
        scene_token = sample_to_scene[sample_token]
        if sample_token not in sample_data_maps[scene_token]:
            # should not happen for keyframes, but be robust
            sample_data_maps[scene_token][sample_token] = {}

        calibrated_sensor = calibrated_sensor_maps[sd["calibrated_sensor_token"]]
        ego_pose = ego_pose_maps[sd["ego_pose_token"]]

        sample_data_maps[scene_token][sample_token]["LIDAR_TOP"] = dict(
            timestamp=sd["timestamp"],
            filename=filename,
            ego_pose=ego_pose,
            calibrated_sensor=calibrated_sensor,
            image_wh=(None, None),
        )

    n_scenes = len(scene_names_sorted)
    n_val = max(1, int(round(n_scenes * float(args.val_ratio))))
    val_names = scene_names_sorted[-n_val:]
    train_names = scene_names_sorted[:-n_val]

    train_tokens = [scene_dict[name] for name in train_names]
    val_tokens = [scene_dict[name] for name in val_names]

    train_metas = {token: sample_data_maps[token] for token in train_tokens}
    val_metas = {token: sample_data_maps[token] for token in val_tokens}

    with open(os.path.join(data_root, "train_metas.pkl"), "wb") as f:
        pickle.dump(train_metas, f)
    with open(os.path.join(data_root, "val_metas.pkl"), "wb") as f:
        pickle.dump(val_metas, f)

    print(f"Wrote {len(train_metas)} train scenes and {len(val_metas)} val scenes to {data_root}")


if __name__ == "__main__":
    main()

