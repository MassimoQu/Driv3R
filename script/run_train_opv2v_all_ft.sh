#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd)/third_party/sam2:${PYTHONPATH:-}"
TRAIN_SIZE=${TRAIN_SIZE:-2000}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-8} python3 train.py \
  --train_dataset "${TRAIN_SIZE} @ OPV2VDataset(split='train', data_root='/media/tsinghua3090/8626b953-db6f-4e02-b531-fceb130612da/home/OPV2V', sequence_length=5, cams=['camera0','camera1','camera2','camera3'], resolution=224, cav_mode='all', max_cav=5, split_mode='random', max_scenes=800, depth_mode='lidar', cache=True)" \
  --test_dataset "OPV2VDataset(split='validate', data_root='/media/tsinghua3090/8626b953-db6f-4e02-b531-fceb130612da/home/OPV2V', sequence_length=5, cams=['camera0','camera1','camera2','camera3'], resolution=224, cav_mode='all', max_cav=5, split_mode='left', depth_mode='lidar', cache=True)" \
  --pretrained ./checkpoints/driv3r.pth \
  --output_dir "./output/opv2v_all_ft_e2_${TRAIN_SIZE}" \
  --epochs 2 \
  --batch_size 2 \
  --batch_size_test 1 \
  --num_workers 6 \
  --num_workers_test 0 \
  --amp 0 \
  --disable_flow
