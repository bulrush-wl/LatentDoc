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

# os.environ["TOKENIZERS_PARALLELISM"] = "false"
from copy import deepcopy
import logging
import pathlib
import torch
import transformers
from easydict import EasyDict as edict

from latentdoc.model.resnet_opt import Rsenet_OPT
from latentdoc.utils.arguments import *
from latentdoc.data import make_supervised_data_module
from latentdoc.train.trainer_vit_fixlr import varyTrainer


DEFAULT_IMAGE_TOKEN = '<image>'
DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'
DEFAULT_IMG_START_TOKEN = '<img>'
DEFAULT_IMG_END_TOKEN = '</img>'
DEFAULT_IM_START_TOKEN = '<|im_start|>'
DEFAULT_IM_END_TOKEN = '<|im_end|>' 
IGNORE_INDEX = -100

special_tokens = {
            'img_token': '<image>',
            'img_patch_token': '<img_patch>',
            'im_start_token': '<|im_start|>',
            'im_end_token': '<|im_end|>' ,
            'img_start_token': '<img>',
            'img_end_token': '</img>',
    }

multimodal_cfg = {
        'img_token_len': 256,
        'model_max_length': 4096,
        'output_attentions': True,
        'output_hidden_states': True,
        'return_dict': True,
        'special_tokens': special_tokens
    }

multimodal_cfg = edict(multimodal_cfg)

def train():

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = Rsenet_OPT(multimodal_cfg)

    img_processor, tokenizer = model.get_img_processor(), model.get_tokenizer()

    dtype = torch.float32
    if training_args.fp16:
        dtype = torch.float16
    if training_args.bf16:
        dtype = torch.bfloat16

    model.to(dtype=dtype, device=training_args.device)

  
    if model_args.freeze_lm_model:
        model.llm.requires_grad_(False)
        for p in model.llm.get_input_embeddings().parameters():
            p.requires_grad = True
    
    if model_args.freeze_vision_encoder:
        model.vision_encoder.requires_grad_(False)

                
    params_grad = [p.numel() for n, p in model.named_parameters() if p.requires_grad]
    print(f"Number of Mapping Trainable Parameters: {sum(params_grad) / (1 << 20):.2f} M")

  
    data_module = make_supervised_data_module(
        data_args=data_args,
        tokenizer=tokenizer, 
        img_processor=img_processor,
        multimodal_cfg = multimodal_cfg
    )

    
    trainer = varyTrainer(
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
