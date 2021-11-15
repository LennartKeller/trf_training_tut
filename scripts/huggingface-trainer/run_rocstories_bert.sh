export CUDA_VISIBLE_DEVICES="0,1"

python rocstories.py \
    --model_name_or_path "bert-base-cased" \
    --output_dir "checkpoints/rocstories/bert" \
    --final_checkpoint_path "final_models/rocstories/bert" \
    --overwrite_output_dir true \
    --learning_rate 3e-5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 \
    --evaluation_strategy "steps" \
    --gradient_accumulation_steps 4 \
    --eval_steps 200 \
    --num_train_epochs 3 \
    --logging_dir "logs/rocstories/bert" \
    --logging_steps 50 \
    --save_strategy "steps" \
    --save_steps 10000 \
    --remove_unused_columns true \
    --logging_first_step true \
    --prediction_loss_only false \
    --seed 42 \
    "$@"