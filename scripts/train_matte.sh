#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 OPM_NUM_THREDDS=8 python train_matte.py 
# CUDA_VISIBLE_DEVICES=0 OPM_NUM_THREDDS=8 python train_matte.py --config '/home/jhb/base/FaceExtraction/config/train2.toml'