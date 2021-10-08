#!/bin/bash
export TOKENIZERS_PARALLELISM=false
python rocstories.py \
	--gradient_clip_val 1.0 \
    --num_processes 10 \
    --val_check_interval 2 \
    --gpus 1 \
	--max_epochs 1 \
    --default_root_dir rocstories \
	"$@"
