
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

from latentdoc.data.conversation import conv_mpt
from latentdoc.data.base_dataset import BaseDataset
# from vary.data.base_dataset import BaseDataset
# from vary.utils.constants import *
# from vary.utils import conversation as conversation_lib

DATASET_INFO = {
    'zhongtie_doc': {'images': "/home/yuhaiyang/zlw/dataset/zhongtie_doc/imgs/",
                     'annotations': "/home/yuhaiyang/zlw/dataset/zhongtie_doc/doc_des_test.json"}
}

DEFAULT_IMAGE_TOKEN = '<image>'
DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'
DEFAULT_IMG_START_TOKEN = '<img>'
DEFAULT_IMG_END_TOKEN = '</img>'

DEFAULT_IM_START_TOKEN = '<|im_start|>'
DEFAULT_IM_END_TOKEN = '<|im_end|>' 
IGNORE_INDEX = -1


class ConversationDataset(BaseDataset):
    """Conversation format dataset stage2 fine-tuning."""

    def __init__(self, datasets, tokenizer, img_processor, multimodal_cfg):
        super(ConversationDataset, self).__init__(datasets, tokenizer, img_processor, multimodal_cfg)
        # v0 version format conversation
        # conversation_lib.default_conversation = conversation_lib.conv_templates["mpt"]
        # logging.warning("Formatting inputs into conversation type: mpt-fixed")
        logging.warning("Loading data...")

        list_data_dict = []
        list_image_path = []

        for name in datasets.split("+"):
            # for name in vary_data_dict[name_all]:
            dataset = DATASET_INFO[name]
            data_path = dataset['annotations']
            data = json.load(open(data_path, "r"))
            list_data_dict.extend(data)
            image_path = dataset['images']
            list_image_path.extend([image_path] * len(data))
            logging.warning(f"Data from {data_path} provide {len(data)} conversations.")

        assert len(list_data_dict) == len(list_image_path)
        logging.warning(f"{len(list_data_dict)} conversations in total.")
        a_new_list = list(zip(list_data_dict, list_image_path))
        random.shuffle(a_new_list)
        list_data_dict_new, list_image_path_new = zip(*a_new_list)
        self.list_data_dict = list_data_dict_new
        self.list_image_path = list_image_path_new

        self.img_patch_token =  tokenizer.encode(self.multimodal_cfg['img_patch_token'], add_special_tokens=False)[0]
        self.img_start_token = tokenizer.encode(self.multimodal_cfg['img_start_token'], add_special_tokens=False)[0]
        self.img_end_token = tokenizer.encode(self.multimodal_cfg['img_end_token'], add_special_tokens=False)[0]


    
    def multimodal_processor(self, sources):
        for source in sources:
            '''
            有bug
            if 0:
                assert DEFAULT_IMAGE_TOKEN in source[0]['value']
                source[0]['value'] = source[0]['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                source[0]['value'] = DEFAULT_IMAGE_TOKEN + conv_mpt.sep + conv_mpt.roles[0] + ": " + source[0]['value']
            '''
            for sentence in source:
                replace_token = DEFAULT_IMAGE_PATCH_TOKEN * self.multimodal_cfg['img_token_len']
                # if self.multimodal_cfg['use_im_start_end']:
                replace_token = DEFAULT_IMG_START_TOKEN + replace_token + DEFAULT_IMG_END_TOKEN
                sentence["value"] = str(sentence["value"]).replace(DEFAULT_IMAGE_TOKEN, replace_token)
        return sources

    def token_process(self, sources, conv):
        
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
        input_ids = self.tokenizer(
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
            total_len = int(target.ne(self.tokenizer.pad_token_id).sum())

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
                round_len = len(self.tokenizer(rou+conv.sep).input_ids) # + len(tokenizer(conv.sep).input_ids)
                # round_len = len(tokenizer(rou).input_ids) + len(tokenizer(conv.sep).input_ids)

                instruction_len = len(self.tokenizer(parts[0]).input_ids)
                # print(f'{i} round_len: {round_len}')
                # print(f'{i} intstuction_len: {instruction_len}')
                # print(f'{i} target_len: {len(target)}')
                target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
                # print(cur_len + instruction_len)
                cur_len += round_len

            # print(target)
            target[cur_len:] = IGNORE_INDEX

            if cur_len < self.tokenizer.model_max_length:
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


    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # data = self.list_data_dict[i]
        data = copy.deepcopy(self.list_data_dict[1])

        if isinstance(data, dict):
            if 'image' in data:
                image_path = self.list_image_path[i]
                image_file = data['image']

                try:
                    image = Image.open(image_path + image_file).convert('RGB')
                except:
                    print(f'cannot identify image file {image_path + image_file}.')
                    return self.__getitem__(0)

                # try:
                # print(type(image))
                # print(11111111)
                image = self.image_processor(images=image, return_tensors="pt")['pixel_values']
                # except:
                #     print(f'image {image_file} are broken or grayscale! we thus select 0-th sample instead!')
                #     return self.__getitem__(0)

            conversations = self.multimodal_processor([data["conversations"]])

        else:
            conversations = [data]

        # align with fastchat & llava here, put the conversation into a list for tokenization
        data_dict = self.token_process(conversations, conv_mpt)
        data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])
        
        if isinstance(data, dict) and 'image' in data:
            data_dict['image'] = image
            
        else:
            crop_size = self.multimodal_cfg['image_processor'].crop_size
            data_dict['image'] = [torch.zeros(3, crop_size['height'], crop_size['width'])]

        return data_dict


