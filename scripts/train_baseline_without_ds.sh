CUDA_LAUNCH_BLOCKING=1 \
CUDA_VISIBLE_DEVICES=0 \
python      /home/yuhaiyang/zlw/LatentDoc/latentdoc/train/train_sam_opt_1024.py   \
            --model_name_or_path /home/yuhaiyang/zlw/pretrained_weight/models--facebook--opt-125m \
            --vision_encoder /home/yuhaiyang/zlw/pretrained_weight/sam_vit_b_01ec64.pth \
            --freeze_vision_encoder True    \
            --freeze_lm_model False      \
            --img_size 1024   \
            --bf16 True                \
            --per_device_eval_batch_size 16  \
            --gradient_accumulation_steps 1     \
            --evaluation_strategy "no"    \
            --save_strategy "steps"    \
            --save_steps 500    \
            --save_total_limit 5   \
            --weight_decay 0.    \
            --warmup_ratio 0.03   \
            --lr_scheduler_type "cosine_with_restarts"    \
            --logging_steps 1 --tf32 True   \
            --model_max_length 2048    \
            --gradient_checkpointing False     \
            --dataloader_num_workers 8      \
            --report_to none       \
            --per_device_train_batch_size 4   \
            --num_train_epochs 5         \
            --learning_rate 5e-3        \
            --datasets  zhongtie_doc    \
            --output_dir /home/yuhaiyang/zlw/LatentDoc/exps/test_sam_opt_1024     \