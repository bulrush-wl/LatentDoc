from easydict import EasyDict as edict

WEIGHT_PATH = {
    'resnet152': '/home/yuhaiyang/zlw/pretrained_weight/Resnet/resnet152-394f9c45.pth',
    'sam_vit_b': '/home/yuhaiyang/zlw/pretrained_weight/sam_vit_b_01ec64.pth',
    'opt_125': '/home/yuhaiyang/zlw/pretrained_weight/models--facebook--opt-125m',
}


DEFAULT_IMAGE_TOKEN = '<image>'
DEFAULT_IMAGE_PATCH_TOKEN = '<img_patch>'
DEFAULT_IMG_START_TOKEN = '<img>'
DEFAULT_IMG_END_TOKEN = '</img>'
DEFAULT_IM_START_TOKEN = '<|im_start|>'
DEFAULT_IM_END_TOKEN = '<|im_end|>' 
IGNORE_INDEX = -100

special_tokens = {
            'img_token': DEFAULT_IMAGE_TOKEN,
            'img_patch_token': DEFAULT_IMAGE_PATCH_TOKEN,
            'im_start_token': DEFAULT_IM_START_TOKEN,
            'im_end_token': DEFAULT_IM_END_TOKEN ,
            'img_start_token': DEFAULT_IMG_START_TOKEN,
            'img_end_token': DEFAULT_IMG_END_TOKEN,
    }

mm_cfg = {
        'img_token_len': 256,
        'model_max_length': 2048,
        'output_attentions': True,
        'output_hidden_states': True,
        'img_size': 1024,
        'vision_encoder': '',
        'return_dict': True,
        'special_tokens': special_tokens
    }

MM_CFG = edict(mm_cfg)



DATASET_INFO = {
    'zhongtie_doc': {
        'images': '/home/yuhaiyang/zlw/dataset/doc/val_imgs/',
        'annotations': '/home/yuhaiyang/zlw/dataset/doc/doc_conv_val.json',
    },
    
    'test': {
        'images': '/home/yuhaiyang/zlw/dataset/Vary-600k/imgs/',
        'annotations': '/home/yuhaiyang/zlw/dataset/Vary-600k/test.json',
    },

    'test_en': {
        'images': '/home/yuhaiyang/zlw/dataset/Vary-600k/imgs/',
        'annotations': '/home/yuhaiyang/zlw/dataset/Vary-600k/test_en.json',
    },

    'test_cn': {
        'images': '/home/yuhaiyang/zlw/dataset/Vary-600k/imgs/',
        'annotations': '/home/yuhaiyang/zlw/dataset/Vary-600k/test_cn.json',
    },

    'test2': {
        'images': '/home/yuhaiyang/zlw/dataset/Vary-600k/imgs/',
        'annotations': '/home/yuhaiyang/zlw/dataset/Vary-600k/test2.json',
    },
    
    'pdf_cn_30k': {
        'images': '/home/fdu02/fdu02_dir/lw/data/vary-600k/data/pdf_data/pdf_cn_30w/',
        'annotations': '/home/fdu02/fdu02_dir/lw/data/vary-600k/pdf_cn_conv_30w_v2.json',
    },

    'pdf_en_30k': {
        'images': '/home/fdu02/fdu02_dir/lw/data/vary-600k/data/pdf_data/pdf_en_30w/',
        'annotations': '/home/fdu02/fdu02_dir/lw/data/vary-600k/pdf_en_conv_30w_v2.json',
    },

    'DocVQA_train':{
        'images': '/home/fdu02/fdu02_dir/lw/data/DocVQA/image/',
        'annotations': '/home/fdu02/fdu02_dir/lw/data/DocVQA/train_conv.json',
    },
    
    'DocVQA_val':{
         'images': '/home/fdu02/fdu02_dir/lw/data/DocVQA/image/',
         'annotations': '/home/fdu02/fdu02_dir/lw/data/DocVQA/val_conv.json',
    }
}