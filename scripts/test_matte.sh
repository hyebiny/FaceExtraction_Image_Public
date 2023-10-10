#!/usr/bin/env bash
# # CJENM-test CelebA-HQ-WO-test


# root='/experiments/2023-09-22-01-28-04'
# CUDA_VISIBLE_DEVICES=0 python test_matte_modified.py --checkpoint '.'$root'/ckpts/epoch_22_best.ckpt' \
#                 --image-dir '/home/jhb/dataset/FM_dataset/test_benchmark_02' \
#                 --image-ext '.jpg'\
#                 --output '.'$root'/test_images' \
#                 --num -1

# root='/experiments/2023-09-22-09-42-01'
# CUDA_VISIBLE_DEVICES=0 python test_matte_modified.py --checkpoint '.'$root'/ckpts/epoch_23_best.ckpt' \
#                 --image-dir '/home/jhb/dataset/FM_dataset/test_benchmark_02' \
#                 --image-ext '.jpg'\
#                 --output '.'$root'/test_images' \
#                 --num -1

root='/experiments/2023-09-20-11-12-49'
python test_matte_modified_video_fa.py --checkpoint '.'$root'/ckpts/epoch_25_best.ckpt' \
                --image-dir '/home/jhb/dataset/FM_dataset/EF0' \
                --image-ext '.jpg'\
                --output '.'$root'/test_images' \
                --num -1 \
                --inpaint

root='/experiments/2023-09-20-11-33-50'
python test_matte_modified_video_fa.py --checkpoint '.'$root'/ckpts/epoch_20_best.ckpt' \
                --image-dir '/home/jhb/dataset/FM_dataset/EF0' \
                --image-ext '.jpg'\
                --output '.'$root'/test_images' \
                --num -1 \
                --inpaint

root='/experiments/2023-09-21-14-13-26'
python test_matte_modified_video_fa.py --checkpoint '.'$root'/ckpts/epoch_21_best.ckpt' \
                --image-dir '/home/jhb/dataset/FM_dataset/EF0' \
                --image-ext '.jpg'\
                --output '.'$root'/test_images' \
                --num -1 \
                --inpaint

root='/experiments/2023-09-21-14-14-59'
python test_matte_modified_video_fa.py --checkpoint '.'$root'/ckpts/epoch_25_best.ckpt' \
                --image-dir '/home/jhb/dataset/FM_dataset/EF0' \
                --image-ext '.jpg'\
                --output '.'$root'/test_images' \
                --num -1 \
                --inpaint


