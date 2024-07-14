DS_SKIP_CUDA_CHECK=1   \
deepspeed   --include "localhost:0" --master_port 29501 /home/fdu02/fdu02_dir/lw/code/LatentDoc/latentdoc/train/train_sam_opt_1024.py   \
            --deepspeed /home/fdu02/fdu02_dir/lw/code/LatentDoc/zero_config/zero0.json \
            --model_name_or_path   '    '             \
            --img_size 512    \
            --img_token_len 64 \
            --freeze_vision_encoder False    \
            --freeze_lm_model False      \
            --resume True \
            --bf16 True                \
            --per_device_eval_batch_size 16  \
            --gradient_accumulation_steps 1     \
            --evaluation_strategy "no"    \
            --save_strategy "steps"    \
            --save_steps 50    \
            --save_total_limit 10   \
            --weight_decay 0.05    \
            --warmup_ratio 0.03*5   \
            --lr_scheduler_type 'cosine_with_restarts' \
            --logging_steps 1 --tf32 True   \
            --model_max_length 2048    \
            --gradient_checkpointing True     \
            --dataloader_num_workers 12      \
            --report_to none       \
            --per_device_train_batch_size 48   \
            --num_train_epochs 10         \
            --learning_rate 5e-4        \
            --datasets  DocVQA_train    \
            --output_dir /home/fdu02/fdu02_dir/lw/exp/exp_name    \