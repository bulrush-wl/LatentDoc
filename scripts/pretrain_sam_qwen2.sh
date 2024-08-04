DS_SKIP_CUDA_CHECK=1   \
deepspeed   --include "localhost:0" --master_port 29502 /home/yuhaiyang/zlw/LatentDoc/latentdoc/train/train_sam_qwen2.py   \
            --deepspeed /home/yuhaiyang/zlw/LatentDoc/zero_config/zero0.json      \
            --model_type  sam_qwen2  \
            --model_name_or_path   /home/yuhaiyang/zlw/LatentDoc/pretrained_weight/models--Qwen--Qwen2-0.5B-Instruct         \
            --vision_encoder    /home/yuhaiyang/zlw/LatentDoc/pretrained_weight/sam_vit_b_01ec64.pth \
            --img_size 512    \
            --img_token_len 64 \
            --freeze_vision_encoder False    \
            --freeze_lm_model False      \
            --resume False \
            --bf16 True                \
            --per_device_eval_batch_size 16  \
            --gradient_accumulation_steps 1     \
            --evaluation_strategy "no"    \
            --save_strategy "steps"    \
            --save_steps 50    \
            --save_total_limit 1   \
            --weight_decay 0.05    \
            --warmup_ratio 0.15   \
            --lr_scheduler_type 'cosine_with_restarts' \
            --logging_steps 1 --tf32 True   \
            --model_max_length 8192    \
            --gradient_checkpointing True     \
            --dataloader_num_workers 12      \
            --report_to none       \
            --per_device_train_batch_size 2   \
            --num_train_epochs 500         \
            --learning_rate 5e-4        \
            --datasets  test2   \
            --output_dir /home/yuhaiyang/zlw/LatentDoc/exps/test_en    \