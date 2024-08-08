from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from easydict import EasyDict as edict
from latentdoc.model.vision_encoder.sam import build_sam_vit_b_1024
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM
from transformers import Cache, DynamicCache
import logging
from torchvision import transforms


'''
qwen2 使用的 sin-cos position embedding
'''

class LatentDocConfig(Qwen2Config):
    model_type = "latentdoc"

class LatentDocQwen2ForCausalLM(Qwen2ForCausalLM):
    config_class = LatentDocConfig

    def __init__(self, config: Qwen2Config):
        super(LatentDocQwen2ForCausalLM, self).__init__(config)

        '''
        self.model
        self.vocab_size
        self.lm_head
        '''

        self.vision_encoder = build_sam_vit_b_1024()
        self.mm_projector = nn.Linear(1024, self.config.hidden_size)


    def _init_mm_projector(self, ):
        std = 1 #self.config.init_std
        self.mm_projector.weight.data.normal_(mean=0.0, std=std)
        if self.mm_projector.bias is not None:
            self.mm_projector.bias.data.zero_()

    def init_multimodal_module(self, tokenizer, mm_cfg=None, resume=False):

        if self.training and not resume:
            print('*'*12 + 'initing multimodal module' + '*'*12)
          
            print('*'*6 + 'init the project' + '*'*6)
            self._init_mm_projector()

            self.config.mm_cfg = mm_cfg

            print('*'*6 + 'resizing the embedding' + '*'*6)
            self._resize_embedding(tokenizer)

            print('*'*6 + 'expanging the max length' + '*'*6)
            self._expand_max_length()

            print('*'*6 + 'reloading the vision ckpy' + '*'*6)
            self._reload_vision_ckpt()

        elif self.training and resume:

            self.config.mm_cfg = mm_cfg

        elif not self.training:

            mm_cfg = self.config.mm_cfg
            self.config.mm_cfg = edict(mm_cfg)
   
          
        return tokenizer, mm_cfg

    def _reload_vision_ckpt(self,):

        if self.config.mm_cfg.vision_encoder is not None and len(self.config.mm_cfg.vision_encoder) != 0:
            # with open(checkpoint, "rb") as f:
            checkpoint = self.config.mm_cfg.vision_encoder
            print(f'reloading vision encoder weight from {checkpoint}')

            state_dict = torch.load(checkpoint)
            state_dict = { k[14:]:v for k, v in state_dict.items() if 'image_encoder' in k}

            missing_keys, unexpected_keys = self.vision_encoder.load_state_dict(state_dict, strict=False)
            
            print(f'missing_keys: {missing_keys}')
            print(f'unexpected_keys: {unexpected_keys}')

    def _expand_max_length(self,):

        if self.config.mm_cfg.model_max_length > self.config.max_position_embeddings:
            '''
            no need to expand
            '''
            raise NotImplementedError
        pass

    def _resize_embedding(self, tokenizer):

        raw_llm_vocab = self.config.vocab_size

        if len(tokenizer) <= raw_llm_vocab:
            print(f'The raw embedding size is {raw_llm_vocab}, the current number of token is {len(tokenizer)}, no need to do the resize for embedding.')

        else:
            num_new_tokens = len(tokenizer) - raw_llm_vocab

            self.resize_token_embeddings(len(tokenizer))  #

            # init the new embedding
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

        img_features = self.vision_encoder(images)  # b, l, c
        
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
            if (cur_input_ids == self.config.mm_cfg.im_start_token_id).sum()+1 != (cur_input_ids == self.config.mm_cfg.im_end_token_id).sum() and self.training:
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

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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

        
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

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
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # Omit tokens covered by past_key_values
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def multimodal_generate(
        self, input_ids, images, inputs_embeds=None, **kwargs
    ):

        img_features = self.embed_images(images)
        input_embeddings = self.embed_tokens(input_ids)
        multimodal_input_embeddings = self.multimodal_process(input_ids, input_embeddings, img_features)

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
    from latentdoc.model.vision_encoder.sam import build_train_transforms
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

    model_name_or_path = '/home/yuhaiyang/zlw/LatentDoc/pretrained_weight/models--Qwen--Qwen2-1.5B-Instruct'

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

    model = LatentDocQwen2ForCausalLM.from_pretrained(model_name_or_path, ignore_mismatched_sizes=True)
    model.train()
    tokenizer, mm_cfg = model.init_multimodal_module(tokenizer, mm_cfg)

    img_processor = build_train_transforms()


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

    model = LatentDocQwen2ForCausalLM.from_pretrained(model_name_or_path)
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
    


# AutoConfig.register("latentdoc", LatentDocConfig)
# AutoModelForCausalLM.register(LatentDocConfig, LatentDocQwen2ForCausalLM)

if __name__ == '__main__':
    input_ids = torch.arange(0, 30).reshape((2,15))
    # print(input_ids)
    labels = input_ids
    images = torch.randn((2,3,1024,1024))
    model, tokenizer, img_processor = build_model()

    res = model(input_ids=input_ids, labels=labels, images=images)

    print(res.loss)
    print(res.logits.shape)
    # print(res.loss)
    # print(res.loss)