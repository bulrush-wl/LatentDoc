

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
reference: https://github.com/Ucas-HaoranWei/Vary/tree/main

git clone https://github.com/bulrush-wl/LatentDoc.git
cd LatentDoc
pip install -e .
```


## Pretrained Weight
- Download the sam_vit_b in [sam_vit_b](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
- Download the OPT-125 in [Huggingface](https://huggingface.co/facebook/opt-125m)
- Download the qwen2 in [Huggingface](https://huggingface.co/collections/Qwen/qwen2-6659360b33528ced941e557f)

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
   deepspeed   --include "localhost:0" --master_port 29502 path/LatentDoc/latentdoc/train/train_sam_opt.py   \
            --deepspeed path/LatentDoc/zero_config/zero0.json     \
            --model_type  sam_opt  \
            --model_name_or_path  path/LatentDoc/pretrained_weight/models--facebook--opt-125m             \
            --vision_encoder    path/LatentDoc/pretrained_weight/sam_vit_b_01ec64.pth \
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
            --model_max_length 2048    \
            --gradient_checkpointing True     \
            --dataloader_num_workers 12      \
            --report_to none       \
            --per_device_train_batch_size 2   \
            --num_train_epochs 5000         \
            --learning_rate 5e-4        \
            --datasets  test2    \
            --output_dir path/LatentDoc/exps/test    \
   ```

## Infer
  The infer details can be found in LatentDoc/latentdoc/infer

