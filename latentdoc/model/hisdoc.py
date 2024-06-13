# Load model directly
from transformers import AutoProcessor, CLIPTextModel, CLIPVisionModel
from PIL import Image
import requests
import inspect

clip_model_name_or_path = '/home/yuhaiyang/zlw/hisdoc/data/models--openai--clip-vit-large-patch14'

processor = AutoProcessor.from_pretrained(clip_model_name_or_path)
vision_model = CLIPVisionModel.from_pretrained(clip_model_name_or_path)
text_model = CLIPTextModel.from_pretrained(clip_model_name_or_path)


url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
img_input = processor(images=image, return_tensors="pt")
text_input = processor(text=["a photo of a cat", "a photo of a dog", 'a photo of a monkey'], return_tensors="pt", padding=True)


img_output = vision_model(**img_input)
text_output = text_model(**text_input)

pooled_img_output = img_output.pooler_output
pooled_text_output = text_output.pooler_output


print(pooled_img_output.shape)
print(pooled_text_output.shape)


# print(dir(model))
# attributes = inspect.getmembers(model, predicate=inspect.ismethod)
# # 遍历属性列表并打印
# for i, attribute in enumerate(attributes):
#     if i != len(attributes)-1:
#         print(attribute)


# inputs = processor(text=["a photo of a cat", "a photo of a dog", 'a photo of a monkey'], images=image, return_tensors="pt", padding=True)
# img_features = model.get_image_features(pixel_values=inputs['pixel_values'])
# text_features = model.get_input_embeddings(input_ids=inputs['input_ids'])

# print(img_features.shape)
# print(text_features.shape)
# print(inputs)
# outputs = model(**inputs)
# logits_per_image = outputs.logits_per_image # this is the image-text similarity score
# probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities

# print(probs)