from latentdoc.model.resnet_opt import Rsenet_OPT

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

ds = SimpleConversationDateset(datasets='zhongtie_doc', tokenizer=tokenizer, multimodal_cfg=multimodal_cfg)
data = ds[0]

data['input_ids'] = data['input_ids'].unsqueeze(dim=0)
data['labels'] = data['labels'].unsqueeze(dim=0)
data['image'] = data['image'].unsqueeze(dim=0)

model = 



