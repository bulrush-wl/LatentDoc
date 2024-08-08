from timm.models.swin_transformer import SwinTransformer
import torch.nn as nn
import torch
class Swin_Transformer(nn.Module):
    def __init__(self,dim, input_resolution, window_size, shift_size,) -> None:
        super().__init__()
        self.dim=dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size