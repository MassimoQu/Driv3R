#!/usr/bin/env bash
set -euo pipefail
MAX_B=${1:-200}
CKPT=${CKPT:-./checkpoints/driv3r.pth}
NUSC_ROOT=${NUSC_ROOT:-/home/qqxluca/vggt_series_4_coop/datasets/nuscenes}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python3 eval.py \
  --dataset "NuSceneDataset(split='val', data_root='${NUSC_ROOT}', sequence_length=5, cams=['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT'], resolution=224, depth_root='', depth_mode='lidar', dynamic=False)" \
  --ckpt_path "${CKPT}" \
  --resolution 224 \
  --sequence_length 5 \
  --device cuda:0 \
  --max_batches ${MAX_B}
