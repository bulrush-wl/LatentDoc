import numpy as np
from timm.models.swin_transformer import SwinTransformer
import torch.nn as nn
import torch
import albumentations as alb
from albumentations.pytorch import ToTensorV2
import cv2, math
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from typing import Optional, Tuple, Type
from functools import partial
class Swin_Transformer_vanilla(nn.Module):
    def __init__(self, input_resolution,encoder_layer, window_size,embed_dim=96) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        self.encoder_layer=encoder_layer
        self.window_size = window_size
        self.embed_dim=embed_dim
        model=SwinTransformer(
            img_size=self.input_resolution,
            depths=self.encoder_layer,
            window_size=self.window_size,
            patch_size=4,
            embed_dim=self.embed_dim,
            num_heads=[4, 8, 16, 32],
            num_classes=0,
        )
        self.patch_embed=model.patch_embed
        self.layers=model.layers
        self.norm=model.norm
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        @param
        x(b,c,h,w)
        @return
        x(b,n,c)
        '''
        x=self.patch_embed(x)
        x=self.layers(x)
        x=self.norm(x)
        return x
    

class Swin_Transformer_without_patch_embedding(nn.Module):
    def __init__(self, input_resolution,encoder_layer, window_size,embed_dim=96) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        self.encoder_layer=encoder_layer
        self.window_size = window_size
        self.embed_dim=embed_dim
        model=SwinTransformer(
            img_size=self.input_resolution,
            depths=self.encoder_layer,
            window_size=self.window_size,
            patch_size=4,
            embed_dim=self.embed_dim,
            num_heads=[4, 8, 16, 32],
            num_classes=0,
        )
        self.patch_embed=model.patch_embed
        self.layers=model.layers
        self.norm=model.norm
        self.net_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.net_3 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1, bias=False)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        @param
        x(b,h,w,c)
        @return
        x(b,n,c)
        '''
        b,h,w,c=x.shape
        x=x.reshape(b,h*w,c)
        x=self.layers(x)
        x=self.norm(x)
        return x

def build_swim_transformer_1024(checkpoint=None):
    return _build_swim_transformer_vanilla(
        input_size=[1024, 1024],
        encoder_layer=[2, 2, 14, 2],
        window_size=8,
        embed_dim=96,
        checkpoint=checkpoint,
    )
def build_swim_transformer_1024_without_patch_embeding(checkpoint=None):
    return _build_swim_transformer_without_patch_embedding(
        input_size=[512, 512],
        encoder_layer=[2, 2, 14, 2],
        window_size=8,
        embed_dim=96,
        checkpoint=checkpoint,
    )
def _build_swim_transformer_vanilla(
    input_size,
    encoder_layer,
    window_size,
    embed_dim,
    checkpoint=None,
):
    image_encoder=Swin_Transformer_vanilla(input_size,encoder_layer,window_size,embed_dim)
    
    if checkpoint is not None:
        # with open(checkpoint, "rb") as f:
        # print(f'loading sam vit from {checkpoint}')
        state_dict = torch.load(checkpoint)

        state_dict = { k[14:]:v for k, v in state_dict.items() if 'image_encoder' in k}

        missing_keys, unexpected_keys = image_encoder.load_state_dict(state_dict, strict=False)
        
        # print(f'missing_keys: {missing_keys}')
        # print(f'unexpected_keys: {unexpected_keys}')
        # image_encoder.load_state_dict({k[19:]: v for k, v in state_dict.items() if 'vision_tower' in k}, strict=True)
        
    return image_encoder

def _build_swim_transformer_without_patch_embedding(
    input_size,
    encoder_layer,
    window_size,
    embed_dim,
    checkpoint=None,
):
    image_encoder=Swin_Transformer_without_patch_embedding(input_size,encoder_layer,window_size,embed_dim)
    
    if checkpoint is not None:
        # with open(checkpoint, "rb") as f:
        # print(f'loading sam vit from {checkpoint}')
        state_dict = torch.load(checkpoint)

        state_dict = { k[14:]:v for k, v in state_dict.items() if 'image_encoder' in k}

        missing_keys, unexpected_keys = image_encoder.load_state_dict(state_dict, strict=False)
        
        # print(f'missing_keys: {missing_keys}')
        # print(f'unexpected_keys: {unexpected_keys}')
        # image_encoder.load_state_dict({k[19:]: v for k, v in state_dict.items() if 'vision_tower' in k}, strict=True)
        
    return image_encoder

def alb_wrapper(transform):
    def f(im):
        return transform(image=np.asarray(im))["image"]

    return f


class Erosion(alb.ImageOnlyTransform):
    """
    Apply erosion operation to an image.

    Erosion is a morphological operation that shrinks the white regions in a binary image.

    Args:
        scale (int or tuple/list of int): The scale or range for the size of the erosion kernel.
            If an integer is provided, a square kernel of that size will be used.
            If a tuple or list is provided, it should contain two integers representing the minimum
            and maximum sizes for the erosion kernel.
        always_apply (bool, optional): Whether to always apply this transformation. Default is False.
        p (float, optional): The probability of applying this transformation. Default is 0.5.

    Returns:
        numpy.ndarray: The transformed image.
    """

    def __init__(self, scale, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        if type(scale) is tuple or type(scale) is list:
            assert len(scale) == 2
            self.scale = scale
        else:
            self.scale = (scale, scale)

    def apply(self, img, **params):
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, tuple(np.random.randint(self.scale[0], self.scale[1], 2))
        )
        img = cv2.erode(img, kernel, iterations=1)
        return img


class Dilation(alb.ImageOnlyTransform):
    """
    Apply dilation operation to an image.

    Dilation is a morphological operation that expands the white regions in a binary image.

    Args:
        scale (int or tuple/list of int): The scale or range for the size of the dilation kernel.
            If an integer is provided, a square kernel of that size will be used.
            If a tuple or list is provided, it should contain two integers representing the minimum
            and maximum sizes for the dilation kernel.
        always_apply (bool, optional): Whether to always apply this transformation. Default is False.
        p (float, optional): The probability of applying this transformation. Default is 0.5.

    Returns:
        numpy.ndarray: The transformed image.
    """

    def __init__(self, scale, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        if type(scale) is tuple or type(scale) is list:
            assert len(scale) == 2
            self.scale = scale
        else:
            self.scale = (scale, scale)

    def apply(self, img, **params):
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, tuple(np.random.randint(self.scale[0], self.scale[1], 2))
        )
        img = cv2.dilate(img, kernel, iterations=1)
        return img


class Bitmap(alb.ImageOnlyTransform):
    """
    Apply a bitmap-style transformation to an image.

    This transformation replaces all pixel values below a certain threshold with a specified value.

    Args:
        value (int, optional): The value to replace pixels below the threshold with. Default is 0.
        lower (int, optional): The threshold value below which pixels will be replaced. Default is 200.
        always_apply (bool, optional): Whether to always apply this transformation. Default is False.
        p (float, optional): The probability of applying this transformation. Default is 0.5.

    Returns:
        numpy.ndarray: The transformed image.
    """

    def __init__(self, value=0, lower=200, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.lower = lower
        self.value = value

    def apply(self, img, **params):
        img = img.copy()
        img[img < self.lower] = self.value
        return img


def build_train_transforms(img_size=1024):
    train_transform = alb_wrapper(
        alb.Compose(
            [
                Bitmap(p=0),
                alb.OneOf([Erosion((2, 3)), Dilation((2, 3))], p=0.02),
                alb.Affine(shear={"x": (0, 3), "y": (-3, 0)}, cval=(255, 255, 255), p=0.03),
                alb.ShiftScaleRotate(
                    shift_limit_x=(0, 0.04),
                    shift_limit_y=(0, 0.03),
                    scale_limit=(-0.15, 0.03),
                    rotate_limit=2,
                    border_mode=0,
                    interpolation=2,
                    value=(255, 255, 255),
                    p=0.03,
                ),
                alb.GridDistortion(
                    distort_limit=0.05,
                    border_mode=0,
                    interpolation=2,
                    value=(255, 255, 255),
                    p=0.04,
                ),
                alb.Compose(
                    [
                        alb.Affine(
                            translate_px=(0, 5), always_apply=True, cval=(255, 255, 255)
                        ),
                        alb.ElasticTransform(
                            p=1,
                            alpha=50,
                            sigma=120 * 0.1,
                            alpha_affine=120 * 0.01,
                            border_mode=0,
                            value=(255, 255, 255),
                        ),
                    ],
                    p=0.04,
                ),
                alb.RandomBrightnessContrast(0.1, 0.1, True, p=0.03),
                alb.ImageCompression(95, p=0.07),
                alb.GaussNoise(20, p=0.08),
                alb.GaussianBlur((3, 3), p=0.03),
                alb.Resize(img_size, img_size),
                alb.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
                ToTensorV2(),
            ]
        )
    )
    return train_transform


def build_test_transforms(img_size=1024):


    test_transform = alb_wrapper(
        alb.Compose(
            [
                alb.Resize(img_size, img_size),
                alb.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
                ToTensorV2(),
            ]
        )
    )
    return test_transform

if __name__=='__main__':
    # input_size=[1024, 1024]
    # encoder_layer=[2, 2, 14, 2]
    # window_size=8
    # model=Swin_Transformer_vanilla(input_size,encoder_layer,window_size)
    # model=build_swim_transformer_1024()
    # img=torch.rand((1,3,1024,1024))
    model=build_swim_transformer_1024_without_patch_embeding()
    img=torch.rand((1,128,128,96))
    print(img.shape)
    print(model(img).shape)

