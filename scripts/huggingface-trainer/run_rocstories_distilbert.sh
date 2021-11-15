export CUDA_VISIBLE_DEVICES="0,1"

python rocstories.py \
    --model_name_or_path "distilbert-base-cased" \
    --output_dir "checkpoints/rocstories/distilbert" \
    --final_checkpoint_path "final_models/rocstories/distilbert" \
    --overwrite_output_dir true \
    --learning_rate 3e-5 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --evaluation_strategy "steps" \
    --gradient_accumulation_steps 1 \
    --eval_steps 200 \
    --num_train_epochs 3 \
    --logging_dir "logs/rocstories/distilbbert" \
    --logging_steps 50 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --remove_unused_columns true \
    --logging_first_step true \
    --prediction_loss_only false \
    --seed 42 \
    "$@"