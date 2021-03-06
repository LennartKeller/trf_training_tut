#!/bin/bash
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="0,2"

python rocstories.py \
    --model.lr 3e-5 \
    --model.model_name_or_path distilbert-base-cased \
    --run.train_batch_size 16 \
    --run.val_batch_size 32 \
    --run.seed 42 \
    --checkpoint.save_last true \
    --checkpoint.mode min \
    --checkpoint.monitor val_loss \
    --checkpoint.every_n_train_steps 10000 \
    --tensorboard.name "rocstories_distilbert" \
    --tensorboard.save_dir "lightning_runs/logs" \
    --trainer.num_processes 10 \
    --trainer.val_check_interval 200 \
    --trainer.log_every_n_steps 50 \
    --trainer.gpus 2 \
    --trainer.max_epochs 3 \
    --trainer.accumulate_grad_batches 1 \
    --trainer.default_root_dir lightning_runs/rocstories \
    --trainer.deterministic true \
    --trainer.accelerator dp \
    "$@"
