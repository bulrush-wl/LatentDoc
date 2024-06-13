from transformers import AutoProcessor, CLIPTextModel, CLIPVisionModel, CLIPVisionModelWithProjection
from PIL import Image
import requests
import inspect
import torch
from transform import train_transform



clip_model_name_or_path = '/home/yuhaiyang/zlw/pretrained_weight/models--openai--clip-vit-large-patch14'

def build_clip_vit_large():
    
    '''
    outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state   # b*257*1024  第一个是cls token
    pooled_output = outputs.pooler_output           # b*1*1024
    '''

    processor = AutoProcessor.from_pretrained(clip_model_name_or_path, size={'shortest_edge':512})
    vision_model = CLIPVisionModel.from_pretrained(clip_model_name_or_path)
    
    return processor, vision_model

def build_clip_vit_with_project():
    processor = AutoProcessor.from_pretrained(clip_model_name_or_path)
    vision_model = CLIPVisionModelWithProjection.from_pretrained(clip_model_name_or_path)
    
    return processor, vision_model

if __name__ == '__main__':

    processor, vision_model = build_clip_vit_large()
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"

    image = Image.open(requests.get(url, stream=True).raw)
    inputs = train_transform(image)
    inputs = inputs.unsqueeze(dim=0)
    print(inputs.shape)
    # inputs = processor(images=image, return_tensors="pt")
    # print(inputs['pixel_values'].shape)
    # img = torch.randn((2,3,512,512))
    outputs = vision_model(inputs)
    image_embeds = outputs.last_hidden_state
    print(image_embeds.shape)
    image_embeds = image_embeds[:,1:,:]
    print(image_embeds.shape)