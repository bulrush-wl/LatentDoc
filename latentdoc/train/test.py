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
from latentdoc.model.resnet_opt_v2 import LatentDocOPTForCausalLM, LatentDocConfig
from latentdoc.model.vision_encoder.resnet import build_transforms

def build_mm_cfg():
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
            'model_max_length': 2048,
            'output_attentions': True,
            'output_hidden_states': True,
            'img_size': 512,
            'return_dict': True,
            'special_tokens': special_tokens
        }

    multimodal_cfg = edict(multimodal_cfg)
    return multimodal_cfg

from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from easydict import EasyDict as edict
from latentdoc.model.llm.opt import build_opt_causal_lm
from latentdoc.model.vision_encoder.resnet import build_resnet152
from transformers import PreTrainedModel
from transformers import OPTConfig, OPTModel, OPTForCausalLM
import logging
from torchvision import transforms


class LatentDocConfig(OPTConfig):
    model_type = "latentdoc"

class LatentDocOPTForCausalLM(PreTrainedModel):
    config_class = LatentDocConfig

    def __init__(self, config: OPTConfig):
        super(LatentDocOPTForCausalLM, self).__init__(config)

        self.net = nn.Linear(2,1)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        images: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,

        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # print('forward')
        # self.check(input_ids, images, labels)
        device = input_ids.device
        a = torch.randn((3,2)).to(device)
        b = self.net(a).squeeze(dim=-1)
        

        return CausalLMOutputWithPast(
            loss= b.sum(),
            logits=None,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )



def train():
    multimodal_cfg = build_mm_cfg()
    
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    multimodal_cfg.model_max_length = training_args.model_max_length
    
    model = LatentDocOPTForCausalLM.from_pretrained(model_args.model_name_or_path)
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=False, padding_side="right", model_max_length=multimodal_cfg.model_max_length)
    img_processor = build_transforms()
    
    model.train()


    for name, param in model.named_parameters():
        param.data = param.data.contiguous()

    dtype = torch.float32
    if training_args.fp16:
        dtype = torch.float16
    if training_args.bf16:
        dtype = torch.bfloat16

    model.to(dtype=dtype, device=training_args.device)

                
    params_grad = [p.numel() for n, p in model.named_parameters() if p.requires_grad]
    print(f"Number of Mapping Trainable Parameters: {sum(params_grad) / (1 << 20):.2f} M")

  
    data_module = make_supervised_data_module(
        data_args=data_args,
        tokenizer=tokenizer, 
        img_processor=img_processor,
        multimodal_cfg = multimodal_cfg
    )

    # ds = data_module['train_dataset']
    # print(ds[0])
    trainer = varyTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module)


    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    if training_args.resume is not None and list(pathlib.Path(training_args.resume).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    trainer._safe_save(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
