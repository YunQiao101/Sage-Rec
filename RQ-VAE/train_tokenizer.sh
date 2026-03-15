#!/bin/bash

# 设置工作目录
export WORK_DIR="/root/autodl-tmp/sage"
cd $WORK_DIR  # 切换到工作目录
python ./RQ-VAE/main.py \
  --device cuda:0 \
  --epochs 5000 \
  --eval_step 500 \
  --data_path ./data/Instruments/Instruments.emb-llama-td.npy\
  --alpha 0.01 \
  --beta 0.0001 \
  --cf_emb_path ./RQ-VAE/ckpt/Instruments-32d-sasrec.pt\
  --ckpt_dir ./checkpoint/\
  --loss_type 'mmd' \
  --batch_size 128 \
  --kmeans_interval 50 \
  --maxe 500 \
  --e_dim 32 \
  --num_workers 4 \
  --layers 2048 1024 512 256 128 64 \
  --num_emb_list 128 128 128 128 \
  --use_swanlab