export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0

DATASET=Instruments
OUTPUT_DIR=./ckpt/$DATASET/

python  ./finetune.py \
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --per_device_batch_size 256 \
    --learning_rate 5e-4 \
    --epochs 200 \
    --index_file .index.epoch10000.alpha1e-1-beta1e-4.json \
    --temperature 1.0 \
    --gradient_accumulation_steps 4\
    --fp16

