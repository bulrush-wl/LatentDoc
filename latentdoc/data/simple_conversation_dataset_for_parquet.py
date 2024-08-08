import io
import os
import copy
import json
import logging
import random
import torch
import transformers
from typing import List, Optional, Tuple, Union, Dict, Sequence
from torch.utils.data import Dataset
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

IGNORE_INDEX = -100
DATASET_INFO = {
    'zhongtie_doc': {
        'images': '/home/yuhaiyang/zlw/dataset/doc/val_imgs/',
        'annotations': '/home/yuhaiyang/zlw/dataset/doc/doc_conv_val.json',
    },
    
    'test': {
        'images': '/home/yuhaiyang/zlw/dataset/Vary-600k/imgs/',
        'annotations': '/home/yuhaiyang/zlw/dataset/Vary-600k/test.json',
    },

    'test2': {
        'images': '/home/yuhaiyang/zlw/dataset/Vary-600k/imgs/',
        'annotations': '/home/yuhaiyang/zlw/dataset/Vary-600k/test2.json',
    }
}


class SimpleConversationDateset(Dataset):
    def __init__(
        self, 
        datasets: str = None,
        tokenizer: transformers.PreTrainedTokenizer = None,
        img_processor = None, # vision encoder对应的img_processor
        multimodal_cfg: dict = None
    ):
        super(SimpleConversationDateset, self).__init__()
        self.tokenizer = tokenizer
        self.image_processor = img_processor
        self.multimodal_cfg = multimodal_cfg

        logging.warning(f"Using {multimodal_cfg['img_token_len']} tokens for representing image")

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
        self.list_data_dict, self.list_image_path  = zip(*a_new_list)

        self.img_token = self.multimodal_cfg.special_tokens.img_token
        self.img_start_token = self.multimodal_cfg.special_tokens.img_start_token
        self.img_patch_token = self.multimodal_cfg.special_tokens.img_patch_token
        self.img_end_token = self.multimodal_cfg.special_tokens.img_end_token
        self.im_start_token = self.multimodal_cfg.special_tokens.im_start_token
        self.im_end_token = self.multimodal_cfg.special_tokens.im_end_token
        self.img_token_len = self.multimodal_cfg.img_token_len


    def image_preprocess(self, imgs):
        
        '''
        The process is detailed in corresponding img_processor
        return  c, h, w
        '''

        return self.image_processor(imgs)  

    def multimodal_process(self, conversations):
        '''
        replace the img_token with img_patch token 
        '''
        assert self.img_token in conversations[0]['value']
        for i in range(len(conversations)):
            if conversations[i]['from'] == 'human':

                if self.img_token not in conversations[i]['value']:  
                    continue

                replace_token = self.img_start_token + self.img_patch_token*self.img_token_len + self.img_end_token
                replace_token += conversations[i]['from'] + ':'
                conversations[i]['value'] = conversations[i]['value'].replace(self.img_token, replace_token)

        return conversations

    def token_preprocess(self, conversations):

        '''
        conv_template: <im_start><image> human: OCR.<im_end> <im_start>gpt:xxxxxxxxx<im_end><im_start><image> human: OCR.<im_end> <im_start>gpt:xxxxxxxxx<im_end>
        '''

        assert len(conversations) % 2 == 0

        rounds = []
        for i in range(0, len(conversations), 2):
            conv0 = conversations[i]
            conv1 = conversations[i+1]
            
            assert 'human' == conv0['from']
            assert 'gpt' == conv1['from']

            r0 = self.im_start_token + conv0['value'] + self.im_end_token
            r1 = self.im_start_token + conv1['from'] + ':' + conv1['value'] + self.im_end_token

            rounds.append(r0+r1)

        # print(rounds)
        # add the begin and end token
        rounds[0] = self.tokenizer.bos_token + rounds[0]
        rounds[-1] = rounds[-1] + self.tokenizer.eos_token 

        # tokenized
        conversation = ''.join(rounds) 

        # print(conversation)
        input_ids = self.tokenizer.encode(  conversation, 
                                            return_tensors="pt",
                                            add_special_tokens=False, 
                                            max_length=self.tokenizer.model_max_length , 
                                            truncation=True)[0]
        
        targets = input_ids.clone()

        # mask targets
        total_len = int(targets.ne(self.tokenizer.pad_token_id).sum())
        sep = self.im_end_token+self.im_start_token+'gpt:'
        cur_len = 0
        for i, rou in enumerate(rounds):

            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break

            parts[0] += sep
            
            round_len = len(self.tokenizer.encode(rou, add_special_tokens=False)) 

            instruction_len = len(self.tokenizer.encode(parts[0], add_special_tokens=False))

            targets[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        
        # print(conversation)
        targets[cur_len:] = IGNORE_INDEX
        if cur_len < self.tokenizer.model_max_length:
                if cur_len != total_len:
                    targets[:] = IGNORE_INDEX
                    print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" (ignored)"
                    )
        elif cur_len >= self.tokenizer.model_max_length:
            # print(11111)
            assert len(input_ids) == self.tokenizer.model_max_length
            assert len(targets) == self.tokenizer.model_max_length
            im_end_token_id = self.tokenizer.convert_tokens_to_ids(self.im_end_token)
            input_ids[-2] = im_end_token_id
            input_ids[-1] = self.tokenizer.eos_token_id
            targets[-2] = im_end_token_id
            targets[-1] = self.tokenizer.eos_token_id

        return dict(
            input_ids=input_ids,
            labels=targets,
        )

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:

        data = copy.deepcopy(self.list_data_dict[i])

        # process conversations
        conversations = self.multimodal_process(data["conversations"])

        data_dict = self.token_preprocess(conversations)
        data_dict = dict(input_ids=data_dict["input_ids"], labels=data_dict["labels"])

        # process image
        image_path = self.list_image_path[i]
        image_file = data['image']

        try:
            image_raw = Image.open(image_path + image_file).convert('RGB')
        except:
            print(f'cannot identify image file {image_path + image_file}.')
            return self.__getitem__(0)

        try:
            image = self.image_preprocess(image_raw)
        except:
            print(f'image {image_file} are broken or grayscale! we thus select 0-th sample instead!')
            return self.__getitem__(0)

        data_dict['images'] = image_raw
        data_dict['raw_data'] = copy.deepcopy(self.list_data_dict[i])



        return data_dict


def test():
    import torch
    from torch import nn 
    import transformers
    from latentdoc.model.vision_encoder.resnet import build_resnet152_and_img_processor
    from easydict import EasyDict as edict

    model_name_or_path = '/home/yuhaiyang/zlw/pretrained_weight/models--facebook--opt-125m'
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False, padding_side="right", model_max_length=300)
    
    
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
        'special_tokens': special_tokens
    }

    tokenizer.add_special_tokens({
        'additional_special_tokens': list(multimodal_cfg['special_tokens'].values())
    })
    # print(tokenizer.convert_tokens_to_ids(tokenizer.bos_token))
    # print(tokenizer.bos_token)
    # print(tokenizer.eos_token)
    # print(tokenizer.pad_token)
    multimodal_cfg = edict(multimodal_cfg)
    img_processor, _ = build_resnet152_and_img_processor()

    ds = SimpleConversationDateset(datasets='zhongtie_doc', img_processor=img_processor, tokenizer=tokenizer, multimodal_cfg=multimodal_cfg)
    data = ds[0]

    # input_ids = data['input_ids']
    # labels = data['labels']
    # img = data['images']

    # print(img.shape)
    # print(input_ids)
    # print(labels)

    # print(tokenizer.decode(input_ids))
    # index = torch.nonzero(labels==-100).squeeze()
    # labels = labels[index[-1]+1:]
    
    # print('labels')
    # print(labels)
    # print(tokenizer.decode(labels))

if __name__ == '__main__':
    test()
    pass