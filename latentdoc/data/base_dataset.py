import io
import os
import copy
import json
import logging
import torch
import transformers
from typing import List, Optional, Tuple, Union, Dict, Sequence
from torch.utils.data import Dataset
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True



def test_img_processor():

    from transformers import AutoProcessor, CLIPTextModel, CLIPVisionModel, CLIPVisionModelWithProjection
    from PIL import Image
    import requests
    import inspect

    clip_model_name_or_path = '/home/yuhaiyang/zlw/hisdoc/data/models--openai--clip-vit-large-patch14'
    processor = AutoProcessor.from_pretrained(clip_model_name_or_path)

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"

    image = Image.open(requests.get(url, stream=True).raw)
    inputs = processor(images=[image, image], return_tensors="pt")
    print(inputs['pixel_values'].shape)



class BaseDataset(Dataset):
    def __init__(
        self, 
        datasets: str = None,
        tokenizer: transformers.PreTrainedTokenizer = None,
        img_processor = None, # vision encoder对应的img_processor
        multimodal_cfg: dict = None
    ):
        super(BaseDataset, self).__init__()
        self.tokenizer = tokenizer
        self.image_processor = img_processor
        self.multimodal_cfg = multimodal_cfg

        logging.warning(f"Using {multimodal_cfg['img_token_len']} tokens for representing image")

    def image_preprocess(self, imgs):
        
        '''
        The process is detailed in corresponding img_processor
        '''

        return self.image_processor(imgs)  


    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        pass