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
# from latentdoc.model.sam_opt_1024_with_ae_with_projector_down4 import LatentDocOPTForCausalLM, LatentDocConfig
# from latentdoc.model.sam_opt_1024 import LatentDocOPTForCausalLM, LatentDocConfig
from transformers import AutoConfig, AutoModel

# from latentdoc.model.vision_encoder.sam import build_test_transforms
from latentdoc.eval.metric.anls import anls_score
from latentdoc.eval.metric.acc import Is_correct

def customer_import(model_type):
    
    global LatentDocOPTForCausalLM, LatentDocConfig, build_test_transforms
    if model_type == 'sam_opt_1024':
        
        from latentdoc.model.sam_opt_1024 import LatentDocOPTForCausalLM, LatentDocConfig
        from latentdoc.model.vision_encoder.sam import build_test_transforms

    elif model_type == 'sam_opt_1024_with_ae':
        from latentdoc.model.sam_opt_1024_with_ae import LatentDocOPTForCausalLM, LatentDocConfig
        from latentdoc.model.AE.ae import build_test_transforms

    elif model_type == 'sam_opt_1024_with_ae_down4':
        from latentdoc.model.sam_opt_1024_with_ae_down4 import LatentDocOPTForCausalLM, LatentDocConfig
        from latentdoc.model.AE.ae import build_test_transforms

    elif model_type == 'sam_opt_1024_with_ae_with_projector_down2':
        from latentdoc.model.sam_opt_1024_with_ae_with_projector_down2 import LatentDocOPTForCausalLM, LatentDocConfig
        from latentdoc.model.AE.ae import build_test_transforms

    elif model_type == 'sam_opt_1024_with_ae_with_projector_down4_recon':
        from latentdoc.model.sam_opt_1024_with_ae_with_projector_down4_recon import LatentDocOPTForCausalLM, LatentDocConfig
        from latentdoc.model.AE.ae import build_test_transforms
    elif model_type == 'sam_opt_1024_with_ae_with_projector_down4_recon_test':
        from latentdoc.model.sam_opt_1024_with_ae_with_projector_down4_recon_test import LatentDocOPTForCausalLM, LatentDocConfig
        from latentdoc.model.AE.ae import build_test_transforms
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

def init_model(model_name_or_path, device='cuda', dtype=torch.bfloat16,with_ae_loss=False,ae_loss_weight=10):
    config=LatentDocConfig.from_pretrained(model_name_or_path)
    if with_ae_loss:
        config.with_ae_loss=with_ae_loss
        config.ae_loss_weight=ae_loss_weight
    model = LatentDocOPTForCausalLM.from_pretrained(model_name_or_path,config=config)
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
            do_sample=False,
            num_beams = 1,
            streamer=streamer,
            max_new_tokens=1024,
            stopping_criteria=[stopping_criteria]
            )
    output = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    output = output.replace(mm_cfg.special_tokens.im_end_token, '')
    output = output.replace(tokenizer.eos_token, '')
    return output

def infer_dataset(model_name_or_path, eval_data, img_root,with_ae_loss=False,ae_loss_weight=10):
    
    # init the model
    model, tokenizer, img_processor, mm_cfg = init_model(model_name_or_path,with_ae_loss,ae_loss_weight)
    img_token = mm_cfg.special_tokens.img_token

    # infer
    pred_res = []
    for i in tqdm.tqdm(range(len(eval_data))):
        item = eval_data[i]
        img_name = item['image']
        img_path = os.path.join(img_root, img_name)
        assert item['conversations'][0]['from'] == 'human'
        assert item['conversations'][1]['from'] == 'gpt'
        prompt = item['conversations'][0]['value'].replace(img_token, '')

        pred = infer_one_img(img_path, prompt, model, tokenizer, img_processor, mm_cfg)

        pred_res.append({
            img_name: {
                'question': item['conversations'][0]['value'],
                'pred': pred,
                'gt': item['conversations'][1]['value']
            }
        })

    return pred_res

def calculate_metric_v1(gt_json_path, pre_json_path):
    with open(gt_json_path, 'r') as f:
        gt_data = json.load(f)

    with open(pre_json_path, 'r') as f:
        pred_data = json.load(f)

    total = 0
    correct = 0

    for gt_item, pred_item in tqdm.tqdm(zip(gt_data, pred_data)):
        assert gt_item['image'] == pred_item['image']

        for gt_conv, pred_conv in zip(gt_item['conversations'], pred_item['conversations']):
            if gt_conv['from'] == 'gpt':
                total += 1
                if relaxed_correctness(gt_conv['value'], pred_conv['value']):
                    correct += 1
    pred_data.append({'acc': correct/total})

    with open(pre_json_path, 'w') as f:
        json.dump(pred_data, f, ensure_ascii=False)

    return total, correct

def calculate_acc(pre_json_path, numeric_toleration=0.5, str_relaxed=False):

    with open(pre_json_path, 'r') as f:
        pred_data = json.load(f)

    total = 0
    correct = 0

    for conversation in tqdm.tqdm(pred_data):
        for _,item in conversation.items():
            gt = item['gt']
            pred = item['pred']
            correct += Is_correct([gt], pred, numeric_toleration, str_relaxed)
            total += 1

    return total, correct 

def calculate_anls(pre_json_path, threshold=0.5):
    with open(pre_json_path, 'r') as f:
        pred_data = json.load(f)

    total = 0
    correct = 0


    for conversation in tqdm.tqdm(pred_data):
        for _,item in conversation.items():
            gt = item['gt']
            pred = item['pred']
            correct += anls_score(pred, [gt], threshold)
            total += 1

    return total, correct 


import os
import fnmatch

def find_test_json_files(root_path):
    """
    递归查找指定目录下所有名为"DocVQA_pred.json"的文件，并返回它们的绝对路径列表。
    
    :param root_path: 查找的根目录
    :return: 包含所有匹配文件绝对路径的列表
    """
    matches = []
    
    for root, dirnames, filenames in os.walk(root_path):
        for filename in fnmatch.filter(filenames, 'DocVQA_pred.json'):
            matches.append(os.path.join(root, filename))
    
    return matches

if __name__ == '__main__':

    # preform customer import
    model_type = 'sam_opt_1024_with_ae_with_projector_down4_recon'
    customer_import(model_type)

    # eval and save the pred
    model_name_or_path = 'exps/recon_test_3/checkpoint-2200'
    eval_json_path = '/home/yuhaiyang/zlw/dataset/Vary-600k/test.json'
    img_root = '/home/yuhaiyang/zlw/dataset/Vary-600k/imgs/'
    temp_res_save_path = os.path.join(model_name_or_path, 'eval_res.json')

    with open(eval_json_path) as f:
        eval_data = json.load(f)
    pred_res = infer_dataset(model_name_or_path, eval_data, img_root)

    # save the pred res
    with open(temp_res_save_path, 'w') as f:
        json.dump(pred_res, f, ensure_ascii=False)
    # eval_DocVQA()

    # calculate the metric
    total, correct=calculate_acc(temp_res_save_path, numeric_toleration=0, str_relaxed=True)
    print('acc in\t', correct/total)
    total, correct=calculate_acc(temp_res_save_path, numeric_toleration=0, str_relaxed=False)
    print('acc ==\t', correct/total)
    total, correct=calculate_anls(temp_res_save_path)
    print('anls\t', correct/total)

    # root_path='/home/fdu02/fdu02_dir/lw/exp/7.9-pretrain-imgsize-1024*1024-sam-opt-epoch5-bs10-ae-lr-5e-5-finetune'
    # json_paths=find_test_json_files(root_path)
    # for json_path in json_paths:
    #     print()
    #     print(json_path)
    #     t,correct=calculate_acc(json_path,numeric_toleration=0,str_relaxed=True)
    #     print('acc in\t',correct/t)
    #     t,correct=calculate_acc(json_path,numeric_toleration=0,str_relaxed=False)
    #     print('acc ==\t',correct/t)
    #     t,correct=calculate_anls(json_path)
    #     print('anls\t',correct/t)