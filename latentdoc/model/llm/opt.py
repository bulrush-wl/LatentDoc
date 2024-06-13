from transformers import AutoTokenizer, OPTForCausalLM, OPTModel
import torch
import torch.nn as nn
import logging
# from torch.nn.parameter import Parameter

opt_model_name_or_path = '/home/yuhaiyang/zlw/pretrained_weight/models--facebook--opt-125m'

class OPTLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # OPT is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, attention_mask: torch.LongTensor, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        attention_mask = attention_mask.long()

        # create positions depending on attention_mask
        positions = (torch.cumsum(attention_mask, dim=1).type_as(attention_mask) * attention_mask).long() - 1

        # cut positions if `past_key_values_length` is > 0
        positions = positions[:, past_key_values_length:]

        return super().forward(positions + self.offset)


def build_opt_causal_lm(config):

    opt_model = OPTForCausalLM.from_pretrained(opt_model_name_or_path)

    if config.model_max_length > opt_model.config.max_position_embeddings:

        # get the original pe weight, nums and dim
        original_pe_weight = opt_model.model.decoder.embed_positions.weight
        num_raw, dim = original_pe_weight.shape[0], original_pe_weight.shape[1]

        # interpolate
        new_pe_weight = interpolate_positional_encoding(original_pe_weight, config.model_max_length)

        # create a new Embedding
        new_embed_positoins = OPTLearnedPositionalEmbedding(config.model_max_length, dim)

        # init the weight
        with torch.no_grad():
            new_embed_positoins.weight.data = new_pe_weight

        # set 
        opt_model.model.decoder.embed_positions = new_embed_positoins

        logging.warning(f'the pe of opt model is interpolated from {opt_model.config.max_position_embeddings} to {config.model_max_length}')
        opt_model.config.max_position_embeddings = config.model_max_length
        opt_model.model.decoder.max_target_positions = config.model_max_length
      

    tokenizer = AutoTokenizer.from_pretrained(opt_model_name_or_path, use_fast=False, padding_side="right", model_max_length=config.model_max_length)   

    return tokenizer, opt_model


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def interpolate_positional_encoding(pe, target_length):
    """
    Interpolates positional encoding from its original length to the target length.

    Args:
        pe (torch.Tensor): The original positional encoding tensor of shape (original_length, d_model).
        target_length (int): The desired length for the positional encoding.

    Returns:
        torch.Tensor: The interpolated positional encoding tensor of shape (target_length, d_model).
    """
    original_length, d_model = pe.shape
    pe = pe.permute(1, 0)
    # Interpolate the positional encoding
    interpolated_pe = F.interpolate(pe.unsqueeze(0), size=(target_length), mode='linear', align_corners=True)
    return interpolated_pe.squeeze(0).permute(1, 0)


def test_interpolate():

    a = nn.Embedding(3, 4)
    print(a.weight.shape)
    # Example usage
    # original_length = 2048
    # target_length = 4
    # d_model = 512  # Assuming the dimensionality of the model

    # Generate a dummy positional encoding for demonstration purposes
    # original_pe = torch.tensor([[2,2,2,2], [3,3,3,2.0]])
    # print(original_pe)
    # Interpolate the positional encoding
    # interpolated_pe = interpolate_positional_encoding(original_pe, target_length)
    # print(interpolated_pe)
    # print("Original PE shape:", original_pe.shape)
    # print("Interpolated PE shape:", interpolated_pe.shape)

if __name__ == '__main__':

    test_interpolate()

    # tokenizer, model = build_opt_causal_lm()

    # print(tokenizer.vocab)

    # prompt = "图中的单位是什么"
    # inputs = tokenizer(prompt, return_tensors="pt")
    # print(inputs)
    # tokenizer.add_tokens('<imgpatch>', special_tokens=True)
    # encode_res = tokenizer.encode('<imgpatch>', add_special_tokens=False)[0]
    # print(type(encode_res))
    # decode_res = tokenizer.decode(inputs.input_ids[0], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    # print(tokenizer.encode('？', add_special_tokens=False))
    # print(decode_res)
    # generate_ids = model.generate(inputs.input_ids, max_length=30)
    # output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    # print(output)