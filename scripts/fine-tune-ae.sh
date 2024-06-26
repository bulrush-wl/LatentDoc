DS_SKIP_CUDA_CHECK=1   \
deepspeed   --include "localhost:4,5,6,7" --master_port 29501 /home/fdu02/fdu02_dir/lw/code/LatentDoc/latentdoc/train/train_sam_opt_1024_with_ae.py   \
            --deepspeed /home/fdu02/fdu02_dir/lw/code/LatentDoc/zero_config/zero0.json \
            --model_name_or_path   /home/fdu02/fdu02_dir/lw/exp/fine-tune_resume_复现vary-sam-opt-1024-V2     \
            --img_size 1024   \
            --freeze_vision_encoder False    \
            --freeze_lm_model False      \
            --freeze_ae False  \
            --bf16 True                \
            --per_device_eval_batch_size 16  \
            --gradient_accumulation_steps 1     \
            --evaluation_strategy "no"    \
            --save_strategy "epoch"    \
            --save_total_limit 10   \
            --weight_decay 0.05    \
            --warmup_ratio 0.03   \
            --lr_scheduler_type 'cosine_with_restarts' \
            --logging_steps 1 --tf32 True   \
            --model_max_length 2048    \
            --gradient_checkpointing True     \
            --dataloader_num_workers 12      \
            --report_to none       \
            --per_device_train_batch_size 16   \
            --num_train_epochs 20        \
            --learning_rate 5e-5        \
            --datasets  DocVQA_train    \
            --output_dir /home/fdu02/fdu02_dir/lw/exp/fine-tune-_resume_fromepoch10  \