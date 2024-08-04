import itertools
import json, sys, os, random
import time, tqdm
from functools import partial
from typing import Optional
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import re, copy
from easydict import EasyDict as edict
import transformers
from transformers import TextStreamer
from latentdoc.utils.utils import disable_torch_init, KeywordsStoppingCriteria

from latentdoc.eval.metric.anls import anls_score
from latentdoc.eval.metric.acc import Is_correct

def customer_import(model_type):
    global LatentDocQwen2ForCausalLM, LatentDocConfig, build_test_transforms

    if model_type == 'sam_qwen2':
        from latentdoc.model.sam_qwen2 import LatentDocQwen2ForCausalLM, LatentDocConfig
        from latentdoc.model.vision_encoder.sam import build_test_transforms
    else:
        print(f'There is no {model_type}')
        exit()  

    

def relaxed_correctness(target: str,
                        prediction: str,
                        max_relative_change: float = 0.05) -> bool:
    """Calculates relaxed correctness.

    The correctness tolerates certain error ratio defined by max_relative_change.
    See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
    “Following Methani et al. (2020), we use a relaxed accuracy measure for the
    numeric answers to allow a minor inaccuracy that may result from the automatic
    data extraction process. We consider an answer to be correct if it is within
    5% of the gold answer. For non-numeric answers, we still need an exact match
    to consider an answer to be correct.”

    Args:
      target: Target string.
      prediction: Predicted string.
      max_relative_change: Maximum relative change.

    Returns:
      Whether the prediction was correct given the specified tolerance.
    """

    def _to_float(text: str) -> Optional[float]:
        try:
            if text.endswith('%'):
                # Convert percentages to floats.
                return float(text.rstrip('%')) / 100.0
            else:
                return float(text)
        except ValueError:
            return None

    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float -
                              target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        # if prediction.lower() in target.lower():
        #     return True
        # else:
        #     return False
        return prediction.lower() == target.lower()

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def init_model(model_name_or_path, device='cuda', dtype=torch.bfloat16):
    model = LatentDocQwen2ForCausalLM.from_pretrained(model_name_or_path)
    model.eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    tokenizer, mm_cfg = model.init_multimodal_module(tokenizer)
    mm_cfg = edict(mm_cfg) 
    img_processor = build_test_transforms(mm_cfg.img_size)
    
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
            max_new_tokens=1024,
            stopping_criteria=[stopping_criteria]
            )
    output = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    output = output.replace(mm_cfg.special_tokens.im_end_token, '')
    output = output.replace(tokenizer.eos_token, '')
    return output


if __name__ == '__main__':

    # preform customer import
    model_type = 'sam_qwen2'
    customer_import(model_type)

    # eval and save the pred
    model_name_or_path = '/home/yuhaiyang/zlw/LatentDoc/exps/test_en/checkpoint-50'
    img_path = '/home/yuhaiyang/zlw/dataset/Vary-600k/imgs/sample.png'
    prompt = '\nRead all the text in the img.'

    # init the model
    model, tokenizer, img_processor, mm_cfg = init_model(model_name_or_path)
    img_token = mm_cfg.special_tokens.img_token

    # infer
    pred = infer_one_img(img_path, prompt, model, tokenizer, img_processor, mm_cfg)
