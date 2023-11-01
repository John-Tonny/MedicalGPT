CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 pretraining.py \
    --model_type chatglm \
    --model_name_or_path /opt/chatglm2-6b \
    --train_file_dir ./data/pretrain \
    --validation_file_dir ./data/pretrain \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --do_eval \
    --use_peft True \
    --seed 42 \
    --fp16 \
    --max_train_samples 10000 \
    --max_eval_samples 10 \
    --num_train_epochs 7.0  \
    --learning_rate 1e-3 \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_steps 50 \
    --evaluation_strategy steps \
    --save_steps 500 \
    --save_strategy steps \
    --save_total_limit 3 \
    --gradient_accumulation_steps 1 \
    --preprocessing_num_workers 1 \
    --block_size 1024 \
    --output_dir outputs-pt-bloom-v1 \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --target_modules all \
    --lora_rank 32 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --torch_dtype float16 \
    --device_map auto \
    --report_to tensorboard \
    --ddp_find_unused_parameters False \
    --gradient_checkpointing True \
    --load_in_8bit False \
    --cache_dir ./cache
