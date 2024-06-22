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
import argparse
import tqdm
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
from copy import deepcopy
import logging
import pathlib
import torch
from PIL import Image
import transformers
from transformers import TextStreamer
from easydict import EasyDict as edict

from latentdoc.utils.utils import disable_torch_init, KeywordsStoppingCriteria
from latentdoc.model.sam_opt_1024 import LatentDocOPTForCausalLM, LatentDocConfig
from latentdoc.model.vision_encoder.sam import build_test_transforms


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def init_model(model_name_or_path, device='cuda', dtype=torch.bfloat16):
    model = LatentDocOPTForCausalLM.from_pretrained(model_name_or_path)
    model.eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    tokenizer, mm_cfg = model.init_multimodal_module(tokenizer)
    mm_cfg = edict(mm_cfg) 
    img_processor = build_test_transforms(mm_cfg.img_size)
    
    # print(model.config)
    # print(type(model.config))
    # print(type(model.config.multimodal_cfg))
    # print(type(mm_cfg))

    return model, tokenizer, img_processor, mm_cfg
   

def infer_one_img(img_path, prompt, model, tokenizer, img_processor, mm_cfg, device='cuda', dtype=torch.bfloat16):

    
    img_token = mm_cfg.special_tokens.img_start_token + mm_cfg.special_tokens.img_patch_token * mm_cfg.img_token_len + mm_cfg.special_tokens.img_end_token
    instruction = img_token + 'human:/n' + prompt
    input_token = mm_cfg.special_tokens.im_start_token + instruction + mm_cfg.special_tokens.im_end_token + mm_cfg.special_tokens.im_start_token + 'gpt:'
    input_token = tokenizer.bos_token + input_token
    input_ids = tokenizer.encode( input_token, 
                                        return_tensors="pt",
                                        add_special_tokens=False, 
                                        max_length=tokenizer.model_max_length, 
                                        truncation=True).to(device=device)
    # print(input_ids.shape)
    img = load_image(img_path) 
    images = img_processor(img).unsqueeze(dim=0).to(device=device, dtype=dtype)
    # print(images.shape)

    stop_str = tokenizer.eos_token
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    model.to(device=device, dtype=dtype)

    with torch.autocast(device, dtype=dtype):
        output_ids = model.multimodal_generate(
            input_ids=input_ids,
            images=images,
            do_sample=True,
            num_beams = 1,
            streamer=streamer,
            max_new_tokens=4096,
            stopping_criteria=[stopping_criteria]
            )
    output = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    output = output.replace(mm_cfg.special_tokens.im_end_token, '')
    output = output.replace(tokenizer.eos_token, '')
    return output
    # print(output_ids)

def infer_from_arg():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_weight_path", type=str, required=False )
    parser.add_argument("--image_file", type=str, required=False )
    parser.add_argument("--prompt", type=str, required=False )
    args = parser.parse_args()

    args.model_weight_path = '/home/yuhaiyang/zlw/LatentDoc/exps/sam_deepspeed_cosine_with_restarts_self/checkpoint-2000'
    args.image_file = '/home/yuhaiyang/zlw/dataset/Vary-600k/imgs/sample_ch.png'
    args.prompt = 'Read all the text in the img.'
    model, tokenizer, img_processor, mm_cfg = init_model(args.model_weight_path)

    infer_one_img(args.image_file, args.prompt, model, tokenizer, img_processor, mm_cfg)
    

if __name__ == "__main__":
    infer_from_arg()
    pass
