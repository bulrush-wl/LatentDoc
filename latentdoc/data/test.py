import io
import os
import copy
import json
import logging
import torch
import random

from typing import List, Optional, Tuple, Union, Dict, Sequence
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from conversation import conv_mpt
from base_dataset import BaseDataset

DEFAULT_IMAGE_TOKEN = '<image>'
DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'
DEFAULT_IMG_START_TOKEN = '<img>'
DEFAULT_IMG_END_TOKEN = '</img>'

DEFAULT_IM_START_TOKEN = '<|im_start|>'
DEFAULT_IM_END_TOKEN = '<|im_end|>' 
IGNORE_INDEX = -1

def multimodal_processor(sources):
        for source in sources:
            '''
            有bug
            if 0:
                assert DEFAULT_IMAGE_TOKEN in source[0]['value']
                source[0]['value'] = source[0]['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                source[0]['value'] = DEFAULT_IMAGE_TOKEN + conv_mpt.sep + conv_mpt.roles[0] + ": " + source[0]['value']
            '''
            for sentence in source:
                replace_token = DEFAULT_IMAGE_PATCH_TOKEN * 256
                # if self.multimodal_cfg['use_im_start_end']:
                replace_token = DEFAULT_IMG_START_TOKEN + replace_token + DEFAULT_IMG_END_TOKEN
                sentence["value"] = str(sentence["value"]).replace(DEFAULT_IMAGE_TOKEN, replace_token)
        return sources

def token_process(sources, conv, tokenizer):
        
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        # Apply prompt templates
        conversations = []
        for i, source in enumerate(sources):
            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]

            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{i}"
                conv.append_message(role, sentence["value"])
            conversations.append(conv.get_prompt())
        # print(conversations)
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=4096,
            truncation=True,
        ).input_ids

        # input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
        targets = input_ids.clone()
        # assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

        # Mask targets
        sep = conv.sep + conv.roles[1]
        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            rounds = conversation.split(conv.sep)
            re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt

            for conv_idx in range(3, len(rounds), 2):
                re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt

            # print(re_rounds)
            cur_len = 0
            target[:cur_len] = IGNORE_INDEX
            for i, rou in enumerate(re_rounds):
                if rou == "":
                    break

                parts = rou.split(sep)
                if len(parts) != 2:
                    break

                parts[0] += sep

                # print(f'{i} rou: {rou}')
                # print(f'{i} parts0: {parts[0]}')
                round_len = len(tokenizer(rou+conv.sep).input_ids) # + len(tokenizer(conv.sep).input_ids)
                # round_len = len(tokenizer(rou).input_ids) + len(tokenizer(conv.sep).input_ids)

                instruction_len = len(tokenizer(parts[0]).input_ids)
                # print(f'{i} round_len: {round_len}')
                # print(f'{i} intstuction_len: {instruction_len}')
                # print(f'{i} target_len: {len(target)}')
                target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
                # print(cur_len + instruction_len)
                cur_len += round_len

            # print(target)
            target[cur_len:] = IGNORE_INDEX

            if cur_len < tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_INDEX
                    print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" (ignored)"
                    )

        return dict(
            input_ids=input_ids,
            labels=targets,
        )

def token_process1(sources, conv, tokenizer):
        
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        # Apply prompt templates
        conversations = []
        for i, source in enumerate(sources):
            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]

            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{i}"
                conv.append_message(role, sentence["value"])
            conversations.append(conv.get_prompt())
        print(conversations)
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=4096,
            truncation=True,
        ).input_ids

        # input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
        targets = input_ids.clone()
        # assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

        # Mask targets
        sep = conv.sep + conv.roles[1]
        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            rounds = conversation.split(conv.sep)
            re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt

            for conv_idx in range(3, len(rounds), 2):
                re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt

            print(re_rounds)
            cur_len = 0
            target[:cur_len] = IGNORE_INDEX
            for i, rou in enumerate(re_rounds):
                if rou == "":
                    break

                parts = rou.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep

                print(f'{i} rou: {rou}')
                print(f'{i} parts0: {parts[0]}')
                round_len = len(tokenizer(rou+conv.sep).input_ids) # + len(tokenizer(conv.sep).input_ids)
                # round_len = len(tokenizer(rou).input_ids) + len(tokenizer(conv.sep).input_ids)

                instruction_len = len(tokenizer(parts[0]).input_ids)
                print(f'{i} round_len: {round_len}')
                print(f'{i} intstuction_len: {instruction_len}')
                print(f'{i} target_len: {len(target)}')
                target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
                print(cur_len + instruction_len)
                cur_len += round_len

            # print(target)
            target[cur_len:] = IGNORE_INDEX

            if cur_len < tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_INDEX
                    
                    print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" (ignored)"
                    )

        return dict(
            input_ids=input_ids,
            labels=targets,
        )


def test1():
    # 添加special tokens
    import torch
    from torch import nn 
    import transformers

    model_name_or_path = '/home/yuhaiyang/zlw/pretrained_weight/models--facebook--opt-125m'
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False, padding_side="right", model_max_length=300)
    
    special_tokens = {
            'img_patch_token': '<img_patch>',
            'im_start_token': '<|im_start|>',
            'im_end_token': '<|im_end|>' ,
            'img_start_token': '<img>',
            'img_end_token': '</img>',
    }

    # print(len(tokenizer))
    num = tokenizer.add_special_tokens({
        'additional_special_tokens': list(special_tokens.values())
    })
    # print(num)
    # print(len(tokenizer))

    print(tokenizer.convert_tokens_to_ids('<img_patch>'))

    num = tokenizer.add_special_tokens({
        'additional_special_tokens': list(special_tokens.values())
    })
    print(tokenizer.convert_tokens_to_ids('<img_patch>'))


if __name__ == '__main__':
    test1()

    
