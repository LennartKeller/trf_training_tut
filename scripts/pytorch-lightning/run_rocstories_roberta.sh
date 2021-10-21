#!/bin/bash
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="0,1,2,3"

python rocstories.py \
	--gradient_clip_val 1.0 \
    --num_processes 10 \
    --val_check_interval 200 \
    --log_every_n_steps 50 \
    --enable_checkpointing true \
    --gpus 1 \
	--max_epochs 3 \
    --default_root_dir lightning_runs/rocstories \
    --lr 3e-5 \
    --deterministic true \
    --seed 42 \
    --model_name_or_path roberta-base \
    --accelerator ddp_spawn \
    --train_batch_size 4 \
    --val_batch_size 8 \
	"$@"
