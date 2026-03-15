DATASET=Instruments
DATA_PATH=../data
OUTPUT_DIR=./ckpt/$DATASET/
CKPT_PATH=./ckpt/$DATASET/

# 定义要跑的 seed
SEEDS=(42 101 2026 8891)

for SEED in "${SEEDS[@]}"; do
    echo "=========================================="
    echo "Running with seed: $SEED"
    echo "=========================================="

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
        --index_file .index.json \
        --model_name_or_path /root/autodl-fs/t5-base \
        --seed $SEED\
        --do_sample \
        --top_k 50 \
        --top_p 0.9 \
        --temperature 0.7
done

echo "All runs completed! Now aggregating results..."
python aggregate_results.py

