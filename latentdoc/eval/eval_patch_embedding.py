import itertools
import json, sys, os, random
import cv2
from matplotlib import pyplot as plt
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
from latentdoc.eval.utils import compute_similarity_matrix, visualize_heatmap, visualize_similarity_matrix
from latentdoc.utils.utils import disable_torch_init, KeywordsStoppingCriteria
from latentdoc.model.sam_opt_1024_with_ae_with_projector_down4_recon import LatentDocOPTForCausalLM, LatentDocConfig
# from latentdoc.model.sam_opt_1024 import LatentDocOPTForCausalLM, LatentDocConfig
from latentdoc.model.AE.ae import build_test_transforms
# from latentdoc.model.vision_encoder.sam import build_test_transforms
from latentdoc.eval.metric.anls import anls_score
from latentdoc.eval.metric.acc import Is_correct
from torchvision import datasets, transforms
import torch.nn as nn
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
    model = LatentDocOPTForCausalLM.from_pretrained(model_name_or_path)
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

def infer_dataset(model_name_or_path, eval_data, img_root):
    
    # init the model
    model, tokenizer, img_processor, mm_cfg = init_model(model_name_or_path)
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


def eval_DocVQA():
    model_name_or_path = r'/home/fdu02/fdu02_dir/zyl/exp/(7-7)input_1024_token_num_64_ae_aefrozen_finetune/'
    json_path = '/home/fdu02/fdu02_dir/lw/data/DocVQA/train_conv.json'
    img_root = '/home/fdu02/fdu02_dir/lw/data/DocVQA/image'
    save_name = 'DocVQA_pred_train.json'

    # 获得当前目录下所有的ckpt
    # ckpt_names = [ dir_name for dir_name in os.listdir(model_name_or_path) if 'checkpoint-' in dir_name]
    ckpt_names = ['checkpoint-1000']

    model_name_or_paths = [model_name_or_path]
    model_name_or_paths += [ os.path.join(model_name_or_path, ckpt_name) for ckpt_name in ckpt_names]

    # 读取数据集json文件
    with open(json_path, 'r') as f:
        eval_data = json.load(f)

    # 对每个ckpt进行预测
    for model_name_or_path in model_name_or_paths:

        # 该ckpt已经预测过
        # if save_name in os.listdir(model_name_or_path):
        #     continue

        # infer
        save_path = os.path.join(model_name_or_path, save_name)
        pred_res = infer_dataset(model_name_or_path, eval_data, img_root)

        # 保存预测结果
        with open(save_path, 'w') as f:
            json.dump(pred_res, f, ensure_ascii=False)
        print(f'The result of prediction is saved in {save_path}')
    
    # 计算评测指标
    
    
    # save_path = f'{model_name_or_path}/DocVQA_pred.json'
    # infer_datasets(model_name_or_path, json_path, img_root, save_path)
    # total, correct = calculate_metric(json_path, save_path)
    # print(f'model name: {model_name_or_path}')
    # print(f'result save path: {save_path}')
    # print(f'total num: {total}, correct num: {correct}, acc: {correct/total}')

import os
import fnmatch

def find_test_json_files(root_path):
    """
    递归查找指定目录下所有名为"test.json"的文件，并返回它们的绝对路径列表。
    
    :param root_path: 查找的根目录
    :return: 包含所有匹配文件绝对路径的列表
    """
    matches = []
    
    for root, dirnames, filenames in os.walk(root_path):
        for filename in fnmatch.filter(filenames, 'DocVQA_pred.json'):
            matches.append(os.path.join(root, filename))
    
    return matches
def get_ae_without_last_bn(ae_encoder):
    ae_encoder_layers=ae_encoder[:-1]
    ae_encoder_last_layer=ae_encoder[-1]
    ae_encoder_last_layer_residual_function=ae_encoder_last_layer.downsample_conv.residual_function[:-1]
    ae_encoder_last_layer_residual_function_bn=ae_encoder_last_layer.downsample_conv.residual_function[-1]
    ae_encoder_last_layer_shortcut=ae_encoder_last_layer.downsample_conv.shortcut[:-1]
    ae_encoder_last_layer_shortcut_bn=ae_encoder_last_layer.downsample_conv.shortcut[-1]
    return ae_encoder_layers,ae_encoder_last_layer_residual_function,ae_encoder_last_layer_shortcut,ae_encoder_last_layer_residual_function_bn,ae_encoder_last_layer_shortcut_bn

def anti_sigmoid(tensor):
    res=torch.exp(tensor)
    return res

if __name__ == '__main__':
    model_name_or_path='exps/recon_test_2/checkpoint-45400'
    # model_name_or_path='exp/ae_noforzen_checkpoint-48500'
    # model_name_or_path='exp/checkpoint-78500'
    device='cuda'
    dtype=torch.bfloat16
    model, tokenizer, img_processor, mm_cfg = init_model(model_name_or_path)
    img_token = mm_cfg.special_tokens.img_token
    # print(model)
    # 获取ae模型及projector参数
    ae_projector=model.ae_projector.to(device=device, dtype=dtype)
    ae_model=model.ae_model.to(device=device, dtype=dtype)
    inc=ae_model.inc.to(device=device, dtype=dtype)
    ae_encoder=ae_model.encoder.to(device=device, dtype=dtype)
    encoder_without_lastbn,ae_encoder_last_layer_residual_function,ae_encoder_last_layer_shortcut,last_layer_residual_function_bn,last_layer_shortcut_bn=get_ae_without_last_bn(ae_encoder)
    outc=ae_model.outc
    decoder=ae_model.decoder
    # #获取patch embedding参数
    # patch_embed=model.vision_encoder.patch_embed.to(device=device, dtype=dtype)
    # print(model.vision_encoder.patch_embed)

    img_path='/home/yuhaiyang/zlw/dataset/Vary-600k/imgs/sample_ch.png'
    img = load_image(img_path) 
    output_name='output.png'
    images = img_processor(img).unsqueeze(dim=0).to(device=device, dtype=dtype)

    # ae
    # encode=ae_projector(ae_encoder(inc(images)).permute(0,2,3,1))

    encode=ae_encoder(inc(images))
    # print(ae_encoder)
    
    
    encode= encode.permute(0,2,3,1)
    encode_without_last_layer= encoder_without_lastbn(inc(images))
    encode_without_last_bn=ae_encoder_last_layer_residual_function(encode_without_last_layer)\
                            +ae_encoder_last_layer_shortcut(encode_without_last_layer)
    encode_after_last_bn=last_layer_residual_function_bn(ae_encoder_last_layer_residual_function(encode_without_last_layer))\
                            +last_layer_shortcut_bn(ae_encoder_last_layer_shortcut(encode_without_last_layer))
    encode_my=nn.ReLU(inplace=True)(encode_after_last_bn)
    output=outc(decoder(encode_my)).squeeze(0).permute(1,2,0).detach().to(torch.float).cpu().numpy()
    encode_my=anti_sigmoid(encode_my).permute(0,2,3,1)
    print(torch.sum(encode-encode_my))
    #patch embedding
    # encode=patch_embed(images)

    print(encode.shape)

    ori_img=cv2.imread(img_path)
    ori_img=cv2.cvtColor(ori_img,cv2.COLOR_BGR2RGB)
    similarity_matrix=compute_similarity_matrix(encode.unsqueeze(0))
    # visualize_similarity_matrix(similarity_matrix)
    # visualize_similarity_matrix(similarity_matrix[755:756,:].repeat(30,1))
    # visualize_heatmap(ori_img,similarity_matrix[200].reshape(32,32).detach().cpu().numpy())

    similarity_matrix_without_lastbn=compute_similarity_matrix(encode_my.unsqueeze(0))
    visualize_similarity_matrix(similarity_matrix_without_lastbn)
    visualize_heatmap(ori_img,similarity_matrix_without_lastbn[200].reshape(32,32).detach().cpu().numpy())
    # print(similarity_matrix)
    # print(similarity_matrix_without_lastbn)

    plt.figure(figsize=(11, 10)) 
    plt.imshow(img)
    plt.gray()
    plt.axis('off')
    plt.savefig(output_name)
    plt.figure(figsize=(11, 10)) 
    plt.imshow(output)
    plt.gray()
    plt.axis('off')
    plt.savefig('output.png')
    # print(model.vision_encoder.patch_embed)
