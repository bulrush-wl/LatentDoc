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
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from copy import deepcopy
import logging
import pathlib
import torch
import transformers


from latentdoc.train.trainer_vit_fixlr import varyTrainer
from latentdoc.model.clip_opt import ClipOPTForCausalLM
from latentdoc.data import make_supervised_data_module
from latentdoc.utils.arguments import *
from latentdoc.utils.utils import smart_tokenizer_and_embedding_resize

DEFAULT_IMAGE_TOKEN = '<image>'
DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'
DEFAULT_IMG_START_TOKEN = '<img>'
DEFAULT_IMG_END_TOKEN = '</img>'
DEFAULT_IM_START_TOKEN = '<|im_start|>'
DEFAULT_IM_END_TOKEN = '<|im_end|>' 
IGNORE_INDEX = -100


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True, padding_side="right", model_max_length=training_args.model_max_length,)

    model = ClipOPTForCausalLM.from_pretrained(model_args.model_name_or_path, low_cpu_mem_usage=True, device_map='cuda')


    multimodal_cfg = {
        'image_token': DEFAULT_IMAGE_TOKEN,
        'img_patch_token': DEFAULT_IMAGE_PATCH_TOKEN,
        'img_start_token': DEFAULT_IMG_START_TOKEN,
        'img_end_token': DEFAULT_IMG_END_TOKEN,
        'im_start_token': DEFAULT_IM_START_TOKEN,
        'im_end_token': DEFAULT_IM_END_TOKEN,
        'img_token_len': 256
    }

    model.initialize_vision_tokenizer(tokenizer, multimodal_cfg)
    img_processor = model.get_model().get_img_processor()['img_processor']

    # smart_tokenizer_and_embedding_resize(
    #     special_tokens_dict=dict(pad_token='<|endoftext|>'),
    #     tokenizer=tokenizer,
    #     model=model,
    #     )

    dtype = torch.float32
    if training_args.fp16:
        dtype = torch.float16
    if training_args.bf16:
        dtype = torch.bfloat16

    # vision_tower_dict = model.get_model().initialize_vision_modules(
    #     vision_tower=model_args.vision_tower,
    #     pretrained_stage1_model=model_args.pretrained_stage1_model,
    #     freeze_vision_tower=model_args.freeze_vision_tower,
    #     use_im_start_end=model_args.use_im_start_end,
    #     vision_select_layer=model_args.vision_select_layer,
    #     dtype=dtype,
    #     device=training_args.device
    # )


    model.to(dtype=dtype, device=training_args.device)

    # data_args.image_token_len = 256
    # data_args.image_processor = vision_tower_dict['image_processor']
    # data_args.image_processor_high = vision_tower_dict['image_processor_high']
    # data_args.use_im_start_end = model_args.use_im_start_end

    # mixed relation, to be fixed
    if model_args.freeze_lm_model:
        model.requires_grad_(False)
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True
        for p in model.get_model().mm_projector_vary.parameters():
            p.requires_grad = True
        for p in model.get_input_embeddings().parameters():
            p.requires_grad = True


    if not model_args.freeze_vision_tower:
        model.get_model().vision_encoder.requires_grad_(True)
    

                
    params_grad = [p.numel() for n, p in model.named_parameters() if p.requires_grad]
    print(f"Number of Mapping Trainable Parameters: {sum(params_grad) / (1 << 20):.2f} M")

  
    data_module = make_supervised_data_module(
        tokenizer=tokenizer, 
        img_processor=img_processor,
        data_args=data_args,
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
        print(1111111111111)
        trainer.train()
    trainer.save_state()
    trainer._safe_save(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
