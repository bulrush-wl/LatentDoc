

# ProjectName

LatentDoc: Understanding the document via Latent Space



<!-- PROJECT LOGO -->
<br />

<p align="center">
  <a href="https://github.com/bulrush-wl/LatentDoc">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">LatentDoc</h3>
  <p align="center">
<!--     LatentDoc -->
<!--     <br /> -->
<!--     <a href="https://github.com/bulrush-wl/LatentDoc"><strong>探索本项目的文档 »</strong></a> -->
<!--     <br /> -->
<!--     <br /> -->
    <a href="https://github.com/bulrush-wl/LatentDoc">Demo</a>
<!--     · -->
<!--     <a href="https://github.com/bulrush-wl/LatentDoc/issues">报告Bug</a> -->
<!--     · -->
<!--     <a href="https://github.com/bulrush-wl/LatentDoc">提出新特性</a> -->
  </p>

</p>



## Contents

- Install
  - Base Environment
  - Package Install
- Download Pretrained Weight
- Train
- Infer



## Install
1. Base Environment
```
  Python 3.8
  torch 2.0.1
```
We suggest that you use the conda environment to install Python and torch in advance.

2. Package Install
- Note: if you have installed Python and torch already, you need to ignore the torch, torchaudio, torchvision in the requirement.txt, manually. Otherwise, the torch, torchaudio, torchvision will be reinstalled again.

```sh
git clone https://github.com/bulrush-wl/LatentDoc.git
cd LatentDoc
pip install -r requirements.txt
pip install -e .
```


## Pretrained Weight
- Download the sam_vit_b in [sam_vit_b](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
- Download the OPT-125 in [Huggingface](https://huggingface.co/facebook/opt-125m)

## Train
1. Add the new dataset
   - the DATASET_INFO config dict can be found in LatentDoc/latentdoc/utils/constant.py
   ```
   DATASET_INFO = {
        'dataset_name': {
            'images': path/img_root/,
            'annotations': path/annotation.json,
        },
    }
   ```
2. Run the training script
   - there are some template scripts in the LatentDoc/scripts
   ```sh
   DS_SKIP_CUDA_CHECK=1   \
   deepspeed   --include "localhost:0,1,2,3" --master_port 29500  /home/yuhaiyang/zyl/code/LatentDoc/latentdoc/train/train_sam_opt_1024_with_ae_recon.py   \
               --deepspeed /home/yuhaiyang/zyl/code/LatentDoc/zero_config/zero0.json \
               --model_type  sam_opt_1024_with_ae_with_projector_down4_recon  \
               --model_name_or_path   /home/yuhaiyang/zyl/code/LatentDoc/pretrained_weight/models--facebook--opt-125m               \
               --vision_encoder    /home/yuhaiyang/zyl/code/LatentDoc/pretrained_weight/sam_vit_b_01ec64.pth \
               --ae /home/yuhaiyang/zyl/code/LatentDoc/pretrained_weight/ae_bestmodel.pth   \
               --with_ae_loss True  \
               --ae_loss_weight 11 \
               --img_size 1024    \
               --img_token_len 64 \
               --freeze_vision_encoder False    \
               --freeze_lm_model False      \
               --freeze_ae False  \
               --is_ae_eval False \
               --resume False \
               --bf16 True                \
               --per_device_eval_batch_size 1  \
               --gradient_accumulation_steps 1     \
               --evaluation_strategy "no"    \
               --save_strategy "steps"    \
               --save_steps 2   \
               --save_total_limit 10   \
               --weight_decay 0.05    \
               --warmup_ratio 0.15   \
               --lr_scheduler_type 'cosine_with_restarts' \
               --logging_steps 1 --tf32 True   \
               --model_max_length 2048    \
               --gradient_checkpointing True     \
               --dataloader_num_workers 12      \
               --report_to none       \
               --per_device_train_batch_size 2  \
               --num_train_epochs 5         \
               --learning_rate 5e-5        \
               --datasets  zhongtie_doc    \
               --output_dir /home/yuhaiyang/zyl/code/LatentDoc/exps/recon_test_4    \
   ```

## Infer
 1、 The LLM infer details can be found in LatentDoc/latentdoc/infer

2、The  Run the ae infer script

there are some template scripts in the LatentDoc/scripts

```sh
python latentdoc/eval/eval_recon.py \
    --model_type sam_opt_1024_with_ae_with_projector_down4_recon \
    --model_name_or_path exps/recon_test_without_ae_pretrain_aeloss_10/checkpoint-96600 \
    --device cuda \
    --img_path /home/yuhaiyang/zlw/dataset/Vary-600k/imgs/sample_ch.png \
    --output_name output.png
```



