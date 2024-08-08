# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import importlib
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
from copy import deepcopy
import logging
import pathlib
import torch
import transformers
from easydict import EasyDict as edict

from latentdoc.utils.constant import MM_CFG
from latentdoc.utils.arguments import *
from latentdoc.data import make_supervised_data_module
from latentdoc.train.latentdoc_trainer import LatentDocTrainer

def customer_import(model_type):
    if model_type == 'sam_qwen2':
        global LatentDocQwen2ForCausalLM, LatentDocConfig, build_train_transforms
        from latentdoc.model.sam_qwen2 import LatentDocQwen2ForCausalLM, LatentDocConfig
        from latentdoc.model.vision_encoder.sam import build_train_transforms
    else:
        print(f'There is no {model_type}')
        exit()  

def init_tokenizer(model_name_or_path, mm_cfg):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False, padding_side="right", model_max_length=mm_cfg.model_max_length )
    num_new_tokens = tokenizer.add_special_tokens({
            'additional_special_tokens': list(mm_cfg.special_tokens.values())
            })

    mm_cfg.img_patch_token_id = tokenizer.convert_tokens_to_ids(mm_cfg.special_tokens.img_patch_token)
    mm_cfg.im_start_token_id = tokenizer.convert_tokens_to_ids(mm_cfg.special_tokens.im_start_token)
    mm_cfg.im_end_token_id = tokenizer.convert_tokens_to_ids(mm_cfg.special_tokens.im_end_token)
    mm_cfg.img_start_token_id = tokenizer.convert_tokens_to_ids(mm_cfg.special_tokens.img_start_token)
    mm_cfg.img_end_token_id = tokenizer.convert_tokens_to_ids(mm_cfg.special_tokens.img_end_token)

    return tokenizer, mm_cfg


def train():
    
    # parse the argument 
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # perform customer import
    customer_import(model_args.model_type) 

    # build and init the mmcfg 
    mm_cfg = deepcopy(MM_CFG)
    mm_cfg.model_max_length = training_args.model_max_length
    mm_cfg.vision_encoder = model_args.vision_encoder
    mm_cfg.img_size = model_args.img_size
    mm_cfg.img_token_len = model_args.img_token_len
    mm_cfg.ae = model_args.ae
    
    # build and init the tokenizer
    tokenizer, mm_cfg = init_tokenizer(model_args.model_name_or_path, mm_cfg)

    # build the img_processor
    img_processor = build_train_transforms(img_size=mm_cfg.img_size)

    # build and init the model
    model = LatentDocQwen2ForCausalLM.from_pretrained(model_args.model_name_or_path)
    model.train()
    tokenizer, mm_cfg = model.init_multimodal_module(tokenizer, mm_cfg)


    dtype = torch.float32
    if training_args.fp16:
        dtype = torch.float16
    if training_args.bf16:
        dtype = torch.bfloat16

    model.to(dtype=dtype, device=training_args.device)

    if model_args.freeze_lm_model:
        model.model.requires_grad_(False)
        for p in model.model.get_input_embeddings().parameters():
            p.requires_grad = True
    
    if model_args.freeze_vision_encoder:
        model.vision_encoder.requires_grad_(False)

    # freeze the ae_model
    if model_args.freeze_ae:
        try:
            model.ae_model.requires_grad_(False)  # 是否要冻住bn
        except:
            print(f'There is no ae model, freeze_ae will not perform.')

    params_grad = [p.numel() for n, p in model.named_parameters() if p.requires_grad]
    print(f"Number of Mapping Trainable Parameters: {sum(params_grad) / (1 << 20):.2f} M")

  
    data_module = make_supervised_data_module(
        data_args=data_args,
        tokenizer=tokenizer, 
        img_processor=img_processor,
        multimodal_cfg = mm_cfg
    )

    # ds = data_module['train_dataset']
    # print(ds[0])
    trainer = LatentDocTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module)


    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
        
    trainer.save_state()
    trainer._safe_save(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
