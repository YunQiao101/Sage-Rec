#!/bin/bash
# 设置工作目录
export WORK_DIR="/root/autodl-tmp/sage"
cd $WORK_DIR  # 切换到工作目录
python ./RQ-VAE/generate_indices.py\
    --dataset Instruments \
    --alpha 1e-1 \
    --beta 1e-4 \
    --epoch 3000\
    --root_path ./checkpoint/Mar-06-2026_14-50-08/\
    --checkpoint epoch_2999_collision_0.9845_model.pth