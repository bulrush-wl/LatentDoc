DS_SKIP_CUDA_CHECK=1   \
deepspeed   --include "localhost:0" --master_port 29500  /home/yuhaiyang/zlw/LatentDoc/latentdoc/train/train_sam_opt_with_ae.py   \
            --deepspeed /home/yuhaiyang/zlw/LatentDoc/zero_config/zero0.json \
            --model_type  sam_opt_1024_with_ae_with_projector_down4  \
            --model_name_or_path   /home/yuhaiyang/zlw/LatentDoc/pretrained_weight/models--facebook--opt-125m               \
            --vision_encoder    /home/yuhaiyang/zlw/LatentDoc/pretrained_weight/sam_vit_b_01ec64.pth \
            --ae  /home/yuhaiyang/zlw/LatentDoc/pretrained_weight/ae_bestmodel.pth   \
            --img_size 1024    \
            --img_token_len 64 \
            --freeze_vision_encoder False    \
            --freeze_lm_model False      \
            --freeze_ae False  \
            --resume False \
            --bf16 True                \
            --per_device_eval_batch_size 1  \
            --gradient_accumulation_steps 1     \
            --evaluation_strategy "no"    \
            --save_strategy "steps"    \
            --save_steps 50    \
            --save_total_limit 1   \
            --weight_decay 0.05    \
            --warmup_ratio 0.15   \
            --lr_scheduler_type 'cosine_with_restarts' \
            --logging_steps 1 --tf32 True   \
            --model_max_length 2048    \
            --gradient_checkpointing True     \
            --dataloader_num_workers 12      \
            --report_to none       \
            --per_device_train_batch_size 1  \
            --num_train_epochs 500         \
            --learning_rate 5e-5        \
            --datasets  test    \
            --output_dir /home/yuhaiyang/zlw/LatentDoc/exps/test    \