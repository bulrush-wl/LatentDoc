from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from easydict import EasyDict as edict
from latentdoc.model.llm.opt import build_opt_causal_lm
from latentdoc.model.vision_encoder.sam_without_patch_embedding import build_sam_vit_b_1024  # without patch embedding
from latentdoc.model.AE.ae import build_ae_model


from transformers import OPTConfig, OPTModel, OPTForCausalLM
import logging
from torchvision import transforms

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

class LatentDocConfig(OPTConfig):
    model_type = "latentdoc"

class LatentDocOPTForCausalLM(OPTForCausalLM):
    config_class = LatentDocConfig

    def __init__(self, config: OPTConfig):
        super(LatentDocOPTForCausalLM, self).__init__(config)

        '''
        self.model
        self.lm_head
        '''
        self.ae_model = build_ae_model()

        self.vision_encoder = build_sam_vit_b_1024()

        self.mm_projector = nn.Linear(1024, self.config.hidden_size)

        self._init_mm_projector()

    def _init_mm_projector(self, ):
        std = self.config.init_std
        self.mm_projector.weight.data.normal_(mean=0.0, std=std)
        if self.mm_projector.bias is not None:
            self.mm_projector.bias.data.zero_()

    def init_multimodal_module(self, tokenizer, mm_cfg=None):

        if self.training:
            print('*'*12 + 'initing multimodal module' + '*'*12)
            # self.num_new_tokens = tokenizer.add_special_tokens({
            # 'additional_special_tokens': list(mm_cfg.special_tokens.values())
            # })

            # mm_cfg.img_patch_token_id = tokenizer.convert_tokens_to_ids(mm_cfg.special_tokens.img_patch_token)
            # mm_cfg.im_start_token_id = tokenizer.convert_tokens_to_ids(mm_cfg.special_tokens.im_start_token)
            # mm_cfg.im_end_token_id = tokenizer.convert_tokens_to_ids(mm_cfg.special_tokens.im_end_token)
            # mm_cfg.img_start_token_id = tokenizer.convert_tokens_to_ids(mm_cfg.special_tokens.img_start_token)
            # mm_cfg.img_end_token_id = tokenizer.convert_tokens_to_ids(mm_cfg.special_tokens.img_end_token)

            self.config.mm_cfg = mm_cfg

            print('*'*6 + 'resizing the embedding' + '*'*6)
            self._resize_embedding(tokenizer)

            print('*'*6 + 'expanging the max length' + '*'*6)
            self._expand_max_length()

            print('*'*6 + 'reloading the vision ckpy' + '*'*6)
            self._reload_vision_ckpt()
        
        else:
            mm_cfg = self.config.mm_cfg
            self.config.mm_cfg = edict(mm_cfg)
          

        return tokenizer, mm_cfg

    def _reload_vision_ckpt(self,):

        # reload the vision encoder weight
        if self.config.mm_cfg.vision_encoder is not None and len(self.config.mm_cfg.vision_encoder) != 0:
            # with open(checkpoint, "rb") as f:
            checkpoint = self.config.mm_cfg.vision_encoder
            print(f'reloading vision encoder weight from {checkpoint}')

            state_dict = torch.load(checkpoint)
            state_dict = { k[14:]:v for k, v in state_dict.items() if 'image_encoder' in k}

            missing_keys, unexpected_keys = self.vision_encoder.load_state_dict(state_dict, strict=False)
            
            print(f'missing_keys: {missing_keys}')
            print(f'unexpected_keys: {unexpected_keys}')

        # reload the ae model weight
        if self.config.mm_cfg.ae is not None and len(self.config.mm_cfg.ae) != 0:

            # with open(checkpoint, "rb") as f:
            checkpoint = self.config.mm_cfg.ae
            print(f'reloading ae model weight from {checkpoint}')

            state_dict = torch.load(checkpoint)
            state_dict = { k.replace('module.', ''):v for k, v in state_dict.items()}

            missing_keys, unexpected_keys = self.ae_model.load_state_dict(state_dict, strict=False)
            
            print(f'missing_keys: {missing_keys}')
            print(f'unexpected_keys: {unexpected_keys}')

    def _expand_max_length(self,):
        if self.config.mm_cfg.model_max_length > self.config.max_position_embeddings:

            # get the original pe weight, nums and dim
            original_pe_weight = self.model.decoder.embed_positions.weight
            num_raw, dim = original_pe_weight.shape[0], original_pe_weight.shape[1]

            # interpolate
            new_pe_weight = interpolate_positional_encoding(original_pe_weight, self.config.mm_cfg.model_max_length + 2)

            # create a new Embedding
            new_embed_positoins = OPTLearnedPositionalEmbedding(self.config.mm_cfg.model_max_length, dim)

            # init the weight
            with torch.no_grad():
                new_embed_positoins.weight.data = new_pe_weight

            # set 
            self.model.decoder.embed_positions = new_embed_positoins

            logging.warning(f'the pe of opt model is interpolated from {self.config.max_position_embeddings} to {self.config.mm_cfg.model_max_length}')
            self.config.max_position_embeddings = self.config.mm_cfg.model_max_length
            self.model.decoder.max_target_positions = self.config.mm_cfg.model_max_length
        pass

    def _resize_embedding(self, tokenizer):

        raw_llm_vocab = self.config.vocab_size

        if len(tokenizer) <= raw_llm_vocab:
            print(f'The raw embedding size is {raw_llm_vocab}, the current number of token is {len(tokenizer)}, no need to do the resize for embedding.')

        else:
            num_new_tokens = len(tokenizer) - raw_llm_vocab

            self.llm.resize_token_embeddings(len(tokenizer))  # do not know

            # resize_token_embeddings will do the following things, too
            input_embeddings = self.get_input_embeddings().weight.data
            output_embeddings = self.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg

            print(f'\nAdded {num_new_tokens} special tokens, the vocab size of llm from {raw_llm_vocab} to {self.config.vocab_size} \n')

    def embed_tokens(self, input_ids):

        return self.get_input_embeddings()(input_ids)


    def embed_images(self, images):

        # add ae encoder
        self.ae_model.eval()
        images = self.ae_model.inc(images)
        images = self.ae_model.encoder(images)
     
        images = images.permute(0, 2, 3, 1)

        # without patch embedding
        img_features = self.vision_encoder(images)  # b, l, c
        
        # print(img_features.shape)

        img_features = self.mm_projector(img_features)

        return img_features

    def multimodal_process(self, input_ids, input_embeddings, img_features):

        dummy_image_features = torch.zeros(256, 1024, device=input_embeddings.device, dtype=input_embeddings.dtype)
        dummy_image_features = self.mm_projector(dummy_image_features)

        new_input_embeds = []
        for cur_input_ids, cur_input_embeds, cur_image_features in zip(input_ids, input_embeddings, img_features):
            # print(cur_input_ids)
            # multimodal LLM, but the current sample is not multimodal
            if (cur_input_ids == self.config.mm_cfg.img_patch_token_id).sum() == 0:
                # print(cur_input_ids)
                # print(self.img_patch_token_id)
                # print(self.img_patch_token_id)
                cur_input_embeds = cur_input_embeds + (0. * dummy_image_features).sum()
                new_input_embeds.append(cur_input_embeds)
                print('pure llm')
                continue

            # check the input
            if (cur_input_ids == self.config.mm_cfg.im_start_token_id).sum() != (cur_input_ids == self.config.mm_cfg.im_end_token_id).sum() and self.training:
                    raise ValueError("The number of input message start tokens and input message end tokens should be the same.")


            if (cur_input_ids == self.config.mm_cfg.img_start_token_id).sum() != (cur_input_ids == self.config.mm_cfg.img_end_token_id).sum() :
                    raise ValueError("The number of image start tokens and image end tokens should be the same.")


            image_start_tokens = torch.where(cur_input_ids == self.config.mm_cfg.img_start_token_id)[0]

            # add the img features to the input embedding, only excute once!
            for image_start_token_pos in image_start_tokens:
                num_patches = cur_image_features.shape[0]

                # check the input
                if cur_input_ids[image_start_token_pos + num_patches + 1] != self.config.mm_cfg.img_end_token_id:
                    raise ValueError("The image end token should follow the image start token.")

                cur_input_embeds = torch.cat(
                        (
                            cur_input_embeds[:image_start_token_pos+1], 
                            cur_image_features, 
                            cur_input_embeds[image_start_token_pos + num_patches + 1:]
                        ), 
                        dim=0
                    )

            new_input_embeds.append(cur_input_embeds)

        inputs_embeds = torch.stack(new_input_embeds, dim=0)
        return inputs_embeds
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        images: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,

        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        # print(self.ae_model.inc.double_conv[0].weight[0,0])
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # print(inputs_embeds)
        # if images is not None:
        # print(images)
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # print(inputs_embeds.shape)
        # print(input_ids.shape)
        if self.training and images is not None:
            img_features = self.embed_images(images)
            inputs_embeds = self.multimodal_process(input_ids, inputs_embeds, img_features)
        # elif images is None and inputs_embeds is not None:
        #     multimodal_input_embeddings = inputs_embeds
        # else:
        #     print(111)
        # else:
        #     raise
        
        outputs = self.model.decoder(
            input_ids=None,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.lm_head(outputs[0]).contiguous()

        # 
        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        # print(input_ids)
        # print('prepare')
        token_type_ids = kwargs.get("token_type_ids", None)
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
            
        else:
            model_inputs = {"input_ids": input_ids}
        # if past_key_values is not None:
        #     print(len(past_key_values))
        #     print(len(past_key_values[0]))
        #     print(len(past_key_values[0][0]))
        #     print(len(past_key_values[0][0][0]))
        #     print(len(past_key_values[0][0][0][0]))
        #     print(inputs_embeds.shape)
        #     print(input_ids.shape)
        # print(model_inputs.keys())
        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                # "images": kwargs.get("images", None),
            }
        )
        return model_inputs

    def multimodal_generate(
        self, input_ids, images, inputs_embeds=None, **kwargs
    ):

        img_features = self.embed_images(images)
        # print(img_features.shape)
        input_embeddings = self.embed_tokens(input_ids)
        multimodal_input_embeddings = self.multimodal_process(input_ids, input_embeddings, img_features)
        # print(input_embeddings.shape)
        # print(multimodal_input_embeddings.shape)

        return self.generate(input_ids=input_ids, inputs_embeds=multimodal_input_embeddings, **kwargs)

    def check(self, input_ids, images, labels):

        print(self.tokenizer.decode(input_ids[0]))
        index = torch.nonzero(labels[0]==-100).squeeze()
        labels = labels[0][index[-1]+1:]
        print(self.tokenizer.decode(labels))

        from torchvision import transforms
        unloader = transforms.ToPILImage()
        image = images.cpu().clone()
        print(image.shape)
        image = image.squeeze(0)
        image = unloader(image)
        image.save('/home/yuhaiyang/zlw/Vary-toy-main/test.png')
        # print(type(input_ids))
        # print(type(labels))
        # print(self.tokenizer.decode(input_ids))
        # print(self.tokenizer.decode(labels))
        

        pass




def build_model():
    import torch
    from torch import nn 
    import transformers
    from latentdoc.data.simple_conversation_dataset import SimpleConversationDateset
    from latentdoc.model.AE.ae import build_train_transform
    from easydict import EasyDict as edict
    
    # build mm_cfg
    special_tokens = {
            'img_token': '<image>',
            'img_patch_token': '<img_patch>',
            'im_start_token': '<|im_start|>',
            'im_end_token': '<|im_end|>' ,
            'img_start_token': '<img>',
            'img_end_token': '</img>',
    }
    mm_cfg = {
        'img_token_len': 256,
        'model_max_length': 2048,
        'output_attentions': True,
        'output_hidden_states': True,
        'return_dict': True,
        'special_tokens': special_tokens
    }
    mm_cfg = edict(mm_cfg)

    model_name_or_path = '/home/yuhaiyang/zlw/LatentDoc/pretrained_weight/models--facebook--opt-125m'

    # build tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False, padding_side="right", model_max_length=mm_cfg.model_max_length )
    num_new_tokens = tokenizer.add_special_tokens({
            'additional_special_tokens': list(mm_cfg.special_tokens.values())
            })

    mm_cfg.img_patch_token_id = tokenizer.convert_tokens_to_ids(mm_cfg.special_tokens.img_patch_token)
    mm_cfg.im_start_token_id = tokenizer.convert_tokens_to_ids(mm_cfg.special_tokens.im_start_token)
    mm_cfg.im_end_token_id = tokenizer.convert_tokens_to_ids(mm_cfg.special_tokens.im_end_token)
    mm_cfg.img_start_token_id = tokenizer.convert_tokens_to_ids(mm_cfg.special_tokens.img_start_token)
    mm_cfg.img_end_token_id = tokenizer.convert_tokens_to_ids(mm_cfg.special_tokens.img_end_token)
    mm_cfg.vision_encoder = '/home/yuhaiyang/zlw/LatentDoc/pretrained_weight/sam_vit_b_01ec64.pth'
    mm_cfg.ae = '/home/yuhaiyang/zlw/LatentDoc/pretrained_weight/ae_bestmodel.pth'

    model = LatentDocOPTForCausalLM.from_pretrained(model_name_or_path, ignore_mismatched_sizes=True)
    model.train()

    tokenizer, mm_cfg = model.init_multimodal_module(tokenizer, mm_cfg)

    img_processor = build_train_transform()

    state_dict = model.state_dict()
    pretrained_state_dict = torch.load('/home/yuhaiyang/zlw/LatentDoc/pretrained_weight/ae_bestmodel.pth')

    # print(state_dict.keys())
    # print(pretrained_state_dict.keys())


    return model, tokenizer, img_processor

def test():
    import torch
    from torch import nn 
    import transformers
    from latentdoc.data.simple_conversation_dataset import SimpleConversationDateset
    from latentdoc.model.vision_encoder.resnet import build_resnet152_and_img_processor
    from easydict import EasyDict as edict
    
    special_tokens = {
            'img_token': '<image>',
            'img_patch_token': '<img_patch>',
            'im_start_token': '<|im_start|>',
            'im_end_token': '<|im_end|>' ,
            'img_start_token': '<img>',
            'img_end_token': '</img>',
    }

    mm_cfg = {
        'img_token_len': 256,
        'model_max_length': 3000,
        'output_attentions': True,
        'output_hidden_states': True,
        'return_dict': True,
        'special_tokens': special_tokens
    }

    mm_cfg = edict(mm_cfg)
    model_name_or_path = '/home/yuhaiyang/zlw/pretrained_weight/models--facebook--opt-125m'

    model = LatentDocOPTForCausalLM.from_pretrained(model_name_or_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False, padding_side="right", model_max_length=2048)
    tokenizer, img_processor = model.init_multimodal_module(tokenizer, mm_cfg)
    
    resnet_path = ''

    # ds = SimpleConversationDateset(datasets='zhongtie_doc', img_processor=img_processor, tokenizer=tokenizer, mm_cfg=mm_cfg)


    # data = ds[0]
    # data['input_ids'] = data['input_ids'].unsqueeze(dim=0)
    # data['labels'] = data['labels'].unsqueeze(dim=0)
    # data['images'] = data['images'].unsqueeze(dim=0)
    # data['input_ids'] = torch.tensor(torch.arange(1,2000), dtype=torch.long).unsqueeze(dim=0)
    # data['labels'] = torch.tensor(torch.arange(1,2000), dtype=torch.long).unsqueeze(dim=0)

    # output = model(**data)
    # print(output.loss)
    pass


def check_model_parameters():
    from latentdoc.model.vision_encoder.sam import build_sam_vit_b_1024

    model, tokenizer, img_processor = build_model()
    model_state_dict = model.state_dict()

    vision_encoder = build_sam_vit_b_1024()
    vision_state_dict = vision_encoder.state_dict()

    true_keys = []
    false_keys = []

    for k, v in model_state_dict.items():
        if 'vision' not in k:
            continue
        v_ = vision_state_dict[k[15:]]

        if (v_ == v).all():
            true_keys.append(k)
        else:
            false_keys.append(k)
        
        # print(f'{k}, {v.size()}')
    print(f'true_keys: {true_keys}')
    print(f'false_keys: {false_keys}')
    

def check_model_parameters_ae():
    

    model, tokenizer, img_processor = build_model()
    model_state_dict = model.state_dict()

    vision_state_dict = torch.load("/home/yuhaiyang/zlw/LatentDoc/pretrained_weight/ae_bestmodel.pth")

    true_keys = []
    false_keys = []

    for k, v in model_state_dict.items():

        if 'ae_model' not in k:
            continue

        v_ = vision_state_dict[k.replace('ae_model', 'module')]

        if (v_ == v).all():
            true_keys.append(k)
        else:
            false_keys.append(k)
        
        # print(f'{k}, {v.size()}')
    print(f'true_keys: {true_keys}')
    print(f'false_keys: {false_keys}')

# AutoConfig.register("latentdoc", LatentDocConfig)
# AutoModelForCausalLM.register(LatentDocConfig, LatentDocOPTForCausalLM)

if __name__ == '__main__':
    from PIL import Image
    model, tokenizer, img_processor = build_model()
    model.cuda()
    img = Image.open('/home/yuhaiyang/zlw/LatentDoc/exps/input.png').convert('RGB')
    img = img_processor(img).unsqueeze(dim=0)
    data = {}
    data['input_ids'] = torch.tensor(torch.arange(1,2000), dtype=torch.long).unsqueeze(dim=0).cuda()
    data['images'] = img.cuda()
    data['labels'] = torch.tensor(torch.arange(1,2000), dtype=torch.long).unsqueeze(dim=0).cuda()
    res = model(**data)
    print(res.loss)
