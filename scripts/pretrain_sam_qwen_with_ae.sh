DS_SKIP_CUDA_CHECK=1   \
deepspeed   --include "localhost:0,1,2,3,4,5" --master_port 29500 /home/yuhaiyang/zlw/LatentDoc/latentdoc/train/train_sam_qwen2_with_ae.py   \
            --deepspeed /home/fdu02/fdu02_dir/lw/code/LatentDoc/zero_config/zero0.json \
            --model_type  sam_qwen2_with_ae  \
            --model_name_or_path   /home/fdu02/fdu02_dir/lw/pretrained_weight/models--facebook--opt-125m               \
            --vision_encoder    /home/fdu02/fdu02_dir/lw/pretrained_weight/sam_vit/sam_vit_b_01ec64.pth \
            --ae /home/fdu02/fdu02_dir/lw/pretrained_weight/ae_bestmodel.pth   \
            --img_size 1024    \
            --img_token_len 64 \
            --freeze_vision_encoder False    \
            --freeze_lm_model False      \
            --freeze_ae True  \
            --resume False \
            --bf16 True                \
            --per_device_eval_batch_size 16  \
            --gradient_accumulation_steps 1     \
            --evaluation_strategy "no"    \
            --save_strategy "steps"    \
            --save_steps 50    \
            --save_total_limit 10   \
            --weight_decay 0.05    \
            --warmup_ratio 0.15   \
            --lr_scheduler_type 'cosine_with_restarts' \
            --logging_steps 1 --tf32 True   \
            --model_max_length 2048    \
            --gradient_checkpointing True     \
            --dataloader_num_workers 12      \
            --report_to none       \
            --per_device_train_batch_size 48   \
            --num_train_epochs 5         \
            --learning_rate 5e-5        \
            --datasets  pdf_cn_30k+pdf_en_30k    \
            --output_dir /home/fdu02/fdu02_dir/lw/exp/    \