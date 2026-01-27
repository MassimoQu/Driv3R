#!/usr/bin/env bash
set -euo pipefail
MAX_B=${1:-200}
CKPT=${CKPT:-./checkpoints/driv3r.pth}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} python3 eval.py \
  --dataset "OPV2VDataset(split='validate', data_root='/media/tsinghua3090/8626b953-db6f-4e02-b531-fceb130612da/home/OPV2V', sequence_length=5, cams=['camera0','camera1','camera2','camera3'], resolution=224, cav_mode='ego', split_mode='left', depth_mode='lidar', cache=True)" \
  --ckpt_path "${CKPT}" \
  --resolution 224 \
  --sequence_length 5 \
  --device cuda:0 \
  --max_batches ${MAX_B}
