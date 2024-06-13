from easydict import EasyDict as edict
special_tokens = {
            'img_token': '<image>',
            'img_patch_token': '<img_patch>',
            'im_start_token': '<|im_start|>',
            'im_end_token': '<|im_end|>' ,
            'img_start_token': '<img>',
            'img_end_token': '</img>',
    }

multimodal_cfg = {
    'img_token_len': 5,
    'special_tokens': special_tokens
}

config = edict(multimodal_cfg)
config.start_token = '<s>'

print(config.start_token)
print(config.keys())
print(config.special_tokens)
