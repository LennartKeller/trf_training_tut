#!/bin/bash
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="0,1,2,3"

python rocstories.py \
    --model.lr 3e-5 \
    --model.model_name_or_path bert-base-cased \
    --run.train_batch_size 4 \
    --run.val_batch_size 8 \
    --run.seed 42 \
    --tensorboard.name "rocstories_roberta" \
    --tensorboard.save_dir "lightning_runs/logs" \
    --trainer.gradient_clip_val 1.0 \
    --trainer.num_processes 10 \
    --trainer.val_check_interval 200 \
    --trainer.log_every_n_steps 50 \
    --trainer.gpus 1 \
    --trainer.max_epochs 3 \
    --trainer.default_root_dir lightning_runs/rocstories \
    --trainer.deterministic true \
    --trainer.accelerator ddp_spawn \
    "$@"
