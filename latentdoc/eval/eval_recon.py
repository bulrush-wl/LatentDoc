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
# from latentdoc.model.sam_opt_1024_with_ae_with_projector_down4_recon import LatentDocOPTForCausalLM, LatentDocConfig
# from latentdoc.model.sam_opt_1024 import LatentDocOPTForCausalLM, LatentDocConfig
# from latentdoc.model.AE.ae import build_test_transforms
# from latentdoc.model.vision_encoder.sam import build_test_transforms
from latentdoc.eval.metric.anls import anls_score
from latentdoc.eval.metric.acc import Is_correct
from torchvision import datasets, transforms
import torch.nn as nn

import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Process some inputs for the model.")
    
    # Add arguments for the model type and model path
    parser.add_argument('--model_type', type=str, default='sam_opt_1024_with_ae_with_projector_down4_recon',
                        help='Type of the model to use.')
    parser.add_argument('--model_name_or_path', type=str, default='exps/recon_test_without_ae_pretrain_aeloss_10/checkpoint-96600',
                        help='Path to the model checkpoint.')
    
    # Add arguments for device and dtype
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run the model on (e.g., "cuda" or "cpu").')

    # Add arguments for image path and output name
    parser.add_argument('--img_path', type=str, default='/home/yuhaiyang/zlw/dataset/Vary-600k/imgs/sample_ch.png',
                        help='Path to the input image.')
    parser.add_argument('--output_name', type=str, default='output.png',
                        help='Name of the output file.')

    args = parser.parse_args()
    

    return args



def customer_import(model_type):

    global LatentDocOPTForCausalLM, LatentDocConfig, build_test_transforms

    if model_type == 'sam_opt_1024_with_ae':
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
    else:
        print(f'There is no {model_type}')
        exit()


def init_model(model_name_or_path, device='cuda', dtype=torch.bfloat16):
    # global build_test_transforms
    model = LatentDocOPTForCausalLM.from_pretrained(model_name_or_path)
    model.eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    tokenizer, mm_cfg = model.init_multimodal_module(tokenizer)
    mm_cfg = edict(mm_cfg) 
    img_processor = build_test_transforms(mm_cfg.img_size)
    
    return model, tokenizer, img_processor, mm_cfg

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image






if __name__ == '__main__':
    args = parse_args()
    customer_import(args.model_type)
    device=args.device
    dtype=torch.bfloat16
    img_path=args.img_path
    output_name=args.output_name
    model, tokenizer, img_processor, mm_cfg = init_model(args.model_name_or_path)
    img_token = mm_cfg.special_tokens.img_token
    ae_projector=model.ae_projector.to(device=device, dtype=dtype)
    ae_model=model.ae_model.to(device=device, dtype=dtype)
    inc=ae_model.inc.to(device=device, dtype=dtype)
    ae_encoder=ae_model.encoder.to(device=device, dtype=dtype)
    outc=ae_model.outc
    decoder=ae_model.decoder
    img = load_image(img_path) 
    images = img_processor(img).unsqueeze(dim=0).to(device=device, dtype=dtype)
    encode=ae_encoder(inc(images))
    output=outc(decoder(encode)).squeeze(0).permute(1,2,0).detach().to(torch.float).cpu().numpy()
    plt.figure(figsize=(11, 10)) 
    plt.imshow(output)
    plt.gray()
    plt.axis('off')
    plt.savefig(output_name ,dpi=300)
