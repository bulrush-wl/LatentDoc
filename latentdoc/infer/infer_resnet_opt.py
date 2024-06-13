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
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
from copy import deepcopy
import logging
import pathlib
import torch
from PIL import Image
import transformers
from transformers import TextStreamer
from easydict import EasyDict as edict

from latentdoc.model.resnet_opt import Rsenet_OPT
from latentdoc.utils.arguments import *
from latentdoc.utils.utils import disable_torch_init, KeywordsStoppingCriteria
from latentdoc.data import make_supervised_data_module
from latentdoc.train.trainer_vit_fixlr import varyTrainer

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
        'return_dict': True,
        'special_tokens': special_tokens
    }

multimodal_cfg = edict(multimodal_cfg)

DEFAULT_IMAGE_TOKEN = multimodal_cfg.special_tokens.img_token
DEFAULT_IMAGE_PATCH_TOKEN = multimodal_cfg.special_tokens.img_patch_token
DEFAULT_IMG_START_TOKEN = multimodal_cfg.special_tokens.img_start_token
DEFAULT_IMG_END_TOKEN = multimodal_cfg.special_tokens.img_end_token
DEFAULT_IM_START_TOKEN = multimodal_cfg.special_tokens.im_start_token
DEFAULT_IM_END_TOKEN = multimodal_cfg.special_tokens.im_end_token
DEFAULT_IMG_TOKEN_LEN = multimodal_cfg.img_token_len
DEFAULT_STOP_TOKEN = '</s>'
IGNORE_INDEX = -100

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def init_model(weight_path, device='cuda', dtype=torch.bfloat16):
    model = Rsenet_OPT(multimodal_cfg)
    # img_processor, tokenizer = model.get_img_processor(), model.get_tokenizer()
    model.load_state_dict(torch.load(weight_path))
    
    model.to(device=device,  dtype=dtype)
    model.eval()

    return model
    # return img_processor, tokenizer, model


def infer_one_img(img_path, prompt, model, device='cuda', dtype=torch.bfloat16):

    
    img_token = DEFAULT_IMG_START_TOKEN + DEFAULT_IMAGE_TOKEN * DEFAULT_IMG_TOKEN_LEN + DEFAULT_IMG_END_TOKEN
    instruction = img_token + 'human:/n' + prompt
    input_token = DEFAULT_IM_START_TOKEN + instruction + DEFAULT_IM_END_TOKEN + DEFAULT_IM_START_TOKEN + 'gpt:'
    input_token = model.tokenizer.bos_token + input_token
    input_ids = model.tokenizer.encode( input_token, 
                                        return_tensors="pt",
                                        add_special_tokens=False, 
                                        max_length=model.tokenizer.model_max_length, 
                                        truncation=True).to(device=device)
    img = load_image(img_path) 
    images = model.img_processor(img).unsqueeze(dim=0).to(device=device, dtype=dtype)
    

    stop_str = DEFAULT_STOP_TOKEN
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, model.tokenizer, input_ids)
    streamer = TextStreamer(model.tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.autocast(device, dtype=dtype):
        output_ids = model.generate(
            input_ids,
            images=images,
            do_sample=True,
            num_beams = 1,
            streamer=streamer,
            max_new_tokens=4096,
            stopping_criteria=[stopping_criteria]
            )



def infer_from_arg():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_weight_path", type=str, required=False )
    parser.add_argument("--image_file", type=str, required=False )
    parser.add_argument("--prompt", type=str, required=False )
    args = parser.parse_args()

    args.model_weight_path = '/home/yuhaiyang/zlw/LatentDoc/exps/test/checkpoint-3150/pytorch_model.bin'
    args.image_file = '/home/yuhaiyang/zlw/dataset/doc/val_imgs/1.pdf_10.png'
    args.prompt = '图中的委托单位是什么？'
    model = init_model(args.model_weight_path)
    infer_one_img(args.image_file, args.prompt, model)
    
    

if __name__ == "__main__":
    infer_from_arg()
    pass
