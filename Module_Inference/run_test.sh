DATASET=Instruments
DATA_PATH=../data
OUTPUT_DIR=./ckpt/$DATASET/
CKPT_PATH=./ckpt/$DATASET/
SEED=(42)
RESULTS_FILE=./results/$DATASET/tiger_seed${SEED}.json
python test.py \
    --gpu_id 0 \
    --ckpt_path $CKPT_PATH \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --results_file $RESULTS_FILE \
    --test_batch_size 32 \
    --num_beams 20 \
    --test_prompt_ids 0 \
    --index_file .index.epoch10000.alpha1e-1-beta1e-4.json \
    --model_name_or_path /root/autodl-fs/t5-base \
    --seed $SEED \


