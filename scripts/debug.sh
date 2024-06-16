CUDA_LAUNCH_BLOCKING=1 \
CUDA_VISIBLE_DEVICES=0 \
python      /home/yuhaiyang/zlw/LatentDoc/latentdoc/train/train_resnet_opt_v3.py   \
            --model_name_or_path /home/yuhaiyang/zlw/pretrained_weight/models--facebook--opt-125m \
            --vision_encoder /home/yuhaiyang/zlw/pretrained_weight/Resnet/resnet152-394f9c45.pth \
            --freeze_vision_encoder False    \
            --freeze_lm_model False      \
            --bf16 True                \
            --per_device_eval_batch_size 16  \
            --gradient_accumulation_steps 1     \
            --evaluation_strategy "no"    \
            --save_strategy "steps"    \
            --save_steps 50    \
            --save_total_limit 5   \
            --weight_decay 0.    \
            --warmup_ratio 0.03   \
            --lr_scheduler_type "cosine"    \
            --logging_steps 1 --tf32 True   \
            --model_max_length 2048    \
            --gradient_checkpointing False     \
            --dataloader_num_workers 8      \
            --report_to none       \
            --per_device_train_batch_size 4   \
            --num_train_epochs 2         \
            --learning_rate 5e-3        \
            --datasets  zhongtie_doc    \
            --output_dir /home/yuhaiyang/zlw/LatentDoc/exps/测试lr-shceduler-修改后的     \