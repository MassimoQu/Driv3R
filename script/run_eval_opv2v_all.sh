#!/usr/bin/env bash
set -euo pipefail
CUDA_VISIBLE_DEVICES=2 python3 eval.py --dataset "OPV2VDataset(split='validate', data_root='/media/tsinghua3090/8626b953-db6f-4e02-b531-fceb130612da/home/OPV2V', sequence_length=5, cams=['camera0','camera1','camera2','camera3'], resolution=224, cav_mode='all', max_cav=5, depth_mode='lidar', cache=True)" --ckpt_path ./checkpoints/driv3r.pth --resolution 224 --sequence_length 5 --device cuda:0
