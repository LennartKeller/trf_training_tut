#!/bin/bash
export TOKENIZERS_PARALLELISM=false
python rocstories.py \
	--gradient_clip_val 1.0 \
    --num_processes 10 \
    --val_check_interval 500 \
    --gpus 1 \
	--max_epochs 3 \
    --default_root_dir lightning_runs/rocstories \
    --lr 5e-5 \
    --deterministic true \
    --seed 42 \
    --model_name_or_path bert-base-cased \
    --accelerator ddp_spawn \
    --train_batch_size 4 \
    --val_batch_size 8 \
	"$@"
