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
import tqdm
import json
from latentdoc.utils.arguments import *
from latentdoc.data.simple_conversation_dataset_for_parquet import SimpleConversationDateset, DATASET_INFO
import base64
import pandas as pd
from PIL import Image
from io import BytesIO
import pyarrow.parquet as pq
import pyarrow as pa
import io
from latentdoc.model.vision_encoder.sam import build_train_transforms

from datasets import load_dataset, concatenate_datasets

def encode_img(img_pil):
    img_byte_arr = io.BytesIO()
    img_pil.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr

def decode_img(img_byte):
    image = Image.open(io.BytesIO(img_byte))
    return image

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

    mm_cfg = {
            'img_token_len': 256,
            'model_max_length': 2048,
            'output_attentions': True,
            'output_hidden_states': True,
            'img_size': 512,
            'return_dict': True,
            'special_tokens': special_tokens
        }

    mm_cfg = edict(mm_cfg)
    return mm_cfg

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

def build_raw_dataset(ds_name, model_name_or_path, img_size, model_max_length, ):
    
    # build and init the mmcfg 
    mm_cfg = build_mm_cfg()
    mm_cfg.model_max_length = model_max_length
    mm_cfg.img_size = img_size
    
    # build and init the tokenizer
    tokenizer, mm_cfg = init_tokenizer(model_name_or_path, mm_cfg)

    # build the img_processor
    img_processor = build_train_transforms(img_size=mm_cfg.img_size)

    ds = SimpleConversationDateset( ds_name,
                                    tokenizer,
                                    img_processor,
                                    mm_cfg)
    return ds

def convert_to_parquet(ds, path_root, file_name, num_size=1024, row_group_size=3000):
    # todo  qa的形式会将图片复制多份

    pq_data = []

    for i in tqdm.tqdm(range(len(ds))):
        item = ds[i]
        images = item['images']
        input_ids = item['input_ids'].numpy().astype(int)
        labels = item['labels'].numpy().astype(int)
        raw_data = item['raw_data']
        img_name = raw_data['image']
        # print(img_name)
        pq_data.append(
            {
                'img_name': img_name,
                'input_ids': input_ids,
                'labels': labels,
                'raw_data': raw_data,
            }
        )
        # exit()
        # if i > 100:
        #     break
    

    num_par = int(len(pq_data) / num_size) + 1
    for i in range(num_par):
        s = i * num_size
        e = min((i+1)*num_size, len(pq_data))
        tmp_pq = pq_data[s:e]
        table = pa.Table.from_pylist(tmp_pq)
        save_path = os.path.join(path_root, f'{file_name}_{i}.parquet')
        print(f"Writing parquet data to => {save_path}")
        pq.write_table(table, save_path, row_group_size=row_group_size)
    
    print('Convert done!')

def build_parquet():
    ds_name = 'zhongtie_doc'
    model_name_or_path = '/home/yuhaiyang/zlw/pretrained_weight/models--facebook--opt-125m'
    img_size = 1024
    model_max_length = 2048
    path_root = '/home/yuhaiyang/zlw/dataset/zhongtie_doc/parquet'
    file_name = 'zhongtie_doc'
    raw_ds = build_raw_dataset(ds_name, model_name_or_path, img_size, model_max_length)
    convert_to_parquet(raw_ds, path_root, file_name)
    
    img_root = DATASET_INFO[ds_name]['images']
    info_path = os.path.join(path_root, 'info.json')
    with open(info_path, 'w') as f:
        json.dump({'total_num': len(raw_ds),
                    'img_root': img_root},
                    f, ensure_ascii=False)

def read_parquet():
    path1 = '/home/yuhaiyang/zlw/dataset/zhongtie_doc/parquet/zhongtie_doc_0.parquet'
    path2 = '/home/yuhaiyang/zlw/dataset/zhongtie_doc/parquet/zhongtie_doc_1.parquet'
    ds1 = load_dataset('parquet', data_files=path1)['train']
    ds2 = load_dataset('parquet', data_files=path2)['train']
    print(len(ds1))
    print(len(ds2))
    ds = concatenate_datasets([ds1, ds2])
    print(len(ds))
    data = ds[0]
    img = data['image']
    input_ids = data['input_ids']
    labels = data['labels']
    raw_data = data['raw_data']

    img = decode_img(img).save('./test.png')
    print(input_ids)
    print(labels)
    print(raw_data)


if __name__ == "__main__":
    build_parquet()
