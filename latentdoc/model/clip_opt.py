


from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM, \
                         CLIPVisionModel, CLIPImageProcessor
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast


# vision encoder
from latentdoc.model.vision_encoder.clip import build_clip_vit_large

# LLM Decoder
from transformers import OPTConfig, OPTModel, OPTForCausalLM

class Clip_OPT(OPTConfig):
    model_type = "clip_opt"


class Clip_OPTModel(OPTModel):
    config_class = Clip_OPT

    def __init__(self, config: Clip_OPT):
        super(Clip_OPTModel, self).__init__(config)


        self.img_processor, self.vision_encoder = build_clip_vit_large()  # 224*224 -> 16*16
        self.mm_projector = nn.Linear(1024, 768)


    def get_img_processor(self, ):

        img_token_len = self.config.img_token_len

        return {'img_processor': self.img_processor,
                'img_token_len': img_token_len}
            

    def embed_tokens(self, x):
        
        return self.get_input_embeddings()(x)


    def encode_img(self, imgs):
        '''
        imgs: tensor

        '''
        img_feature = self.vision_encoder(imgs).last_hidden_state  # b*c*224*224 -> b*257*1024
        
        return img_feature[:,1:,:]   # b*257*1024  -> b*256*1024


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        images: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,

        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        # HACK: replace back original embeddings for LLaVA pretraining
        # orig_embeds_params = getattr(self, 'orig_embeds_params', None)
        # if orig_embeds_params is not None:
        #     with torch.no_grad():
        #         self.get_input_embeddings().weight[:-self.num_new_tokens] = orig_embeds_params[:-self.num_new_tokens].data

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        if images is not None:
            # Multimodal mode
            img_features = self.encode_img(images)  # b*256*1024
            img_features = self.mm_projector(img_features)
        else:
            # LLM mode
            b, l, c = inputs_embeds.shape
            img_features = torch.zeros((b, 256, c))


        # 将原来的inputs中的img_pad_token替换成真正的img_token
        img_patch_token = self.config.img_patch_token
        img_start_token = self.config.img_start_token
        img_end_token = self.config.img_end_token
        use_img_start_end = self.config.use_img_start_end 
        new_input_embeds = []
        for cur_input_ids, cur_input_embeds, cur_image_features in zip(input_ids, inputs_embeds, img_features):

            if (cur_input_ids == img_patch_token).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                new_input_embeds.append(cur_input_embeds)
                continue

            if use_img_start_end:
                if (cur_input_ids == img_start_token).sum() != (cur_input_ids == img_end_token).sum():
                    raise ValueError("The number of image start tokens and image end tokens should be the same.")
                
                image_start_token_pos = torch.where(cur_input_ids == img_start_token)[0]
                num_patches = cur_image_features.shape[0]
                if cur_input_ids[image_start_token_pos + num_patches + 1] != img_end_token:
                        raise ValueError("The image end token should follow the image start token or the number of img_patchs is wrong!.")

                cur_input_embeds = torch.cat(
                        (
                            cur_input_embeds[:image_start_token_pos+1], 
                            cur_image_features, 
                            cur_input_embeds[image_start_token_pos + num_patches + 1:]
                        ), 
                        dim=0
                    )

                new_input_embeds.append(cur_input_embeds)
            else:
                raise NotImplementedError

        inputs_embeds = torch.stack(new_input_embeds, dim=0)  # b * l * c


        return super(Clip_OPTModel, self).forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )


class ClipOPTForCausalLM(OPTForCausalLM):
    config_class = Clip_OPT
    # supports_gradient_checkpointing = True

    def __init__(self, config):
        super(ClipOPTForCausalLM, self).__init__(config)
        self.model = Clip_OPTModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def initialize_vision_tokenizer(
        self, 
        tokenizer, 
        multimodal_cfg,
        freeze_lm_model=False, 
        pretrained_stage1_model=None,
        device="cuda"
    ):
        # 添加vision相关的token
        tokenizer.add_tokens(multimodal_cfg['img_patch_token'], special_tokens=True)  # image patch token 
        tokenizer.add_tokens(multimodal_cfg['img_start_token'], special_tokens=True)  # image start token
        tokenizer.add_tokens(multimodal_cfg['img_end_token'], special_tokens=True)    # image end token
        tokenizer.add_tokens(multimodal_cfg['im_start_token'], special_tokens=True)    # image end token
        tokenizer.add_tokens(multimodal_cfg['im_end_token'], special_tokens=True)    # image end token

        self.resize_token_embeddings(len(tokenizer))

        # 添加特殊token到config中
        self.config.img_patch_token = tokenizer.convert_tokens_to_ids(multimodal_cfg['img_patch_token'])
        self.config.img_start_token = tokenizer.convert_tokens_to_ids(multimodal_cfg['img_start_token'])
        self.config.img_end_token = tokenizer.convert_tokens_to_ids(multimodal_cfg['img_end_token'])
        self.config.use_img_start_end = True
        self.config.img_token_len = multimodal_cfg['img_token_len']

        # self.config.vocab_size = len(tokenizer)

        num_new_tokens = 5
        if num_new_tokens > 0:
            # print(self.get_output_embeddings())
            input_embeddings = self.get_input_embeddings().weight.data
            output_embeddings = self.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg


    def get_model(self):
        return self.model

    # def _set_gradient_checkpointing(self, module, value=False):
    #     if isinstance(module, varyQwenModel):
    #         module.gradient_checkpointing = value


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        images: Optional[torch.FloatTensor] = None,

        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
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

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        # print(input_ids)
        # print(len(images))

        outputs = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            images=images,

            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )


        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states).contiguous()

        # logits
        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()


            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=-1)

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

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

    # def initialize_vision_tokenizer(
    #     self, 
    #     tokenizer, 
    #     freeze_lm_model=False, 
    #     pretrained_stage1_model=None,
    #     device="cuda"
    # ):
    #     config = self.get_model().config

    #     # add image patch token <image>
    #     tokenizer.add_tokens("</s>", special_tokens=True)
    #     self.resize_token_embeddings(len(tokenizer))

    #     tokenizer.add_tokens(DEFAULT_IMAGE_PATCH_TOKEN, special_tokens=True)
    #     self.resize_token_embeddings(len(tokenizer))
    #     config.im_patch_token = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_PATCH_TOKEN)


    #     config.use_im_start_end = True

    #     # add image start token <im_start> and end token <im_end>
    #     if config.use_im_start_end:
    #         num_new_tokens = 2
    #         tokenizer.add_tokens(DEFAULT_IM_START_TOKEN , special_tokens=True)
    #         tokenizer.add_tokens(DEFAULT_IM_END_TOKEN , special_tokens=True)
    #         self.resize_token_embeddings(len(tokenizer))
    #         config.im_start_token = tokenizer.convert_tokens_to_ids(DEFAULT_IM_START_TOKEN)
    #         config.im_end_token =  tokenizer.convert_tokens_to_ids(DEFAULT_IM_END_TOKEN)

    #         # config.im_start_token, config.im_end_token = 151857, 151858

    #         if num_new_tokens > 0:
    #             input_embeddings = self.get_input_embeddings().weight.data
    #             output_embeddings = self.get_output_embeddings().weight.data

    #             input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
    #             output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

    #             input_embeddings[-num_new_tokens:] = input_embeddings_avg
    #             output_embeddings[-num_new_tokens:] = output_embeddings_avg



# AutoConfig.register("vary", varyConfig)
# AutoModelForCausalLM.register(varyConfig, varyOPTForCausalLM)


def test_llm():
    from transformers import AutoTokenizer, OPTForCausalLM, OPTModel
    model_name_or_path = '/home/yuhaiyang/zlw/hisdoc/data/models--facebook--opt-125m'
    model = ClipOPTForCausalLM.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)   

    multimodal_cfg = {
        'img_patch_token': '<imgpatch>',
        'img_start_token': '<img>',
        'img_end_token': '</img>',
        'img_token_len': 256
    }

    model.initialize_vision_tokenizer(tokenizer, multimodal_cfg)

    prompt = 'Hello, Who are you?'
    inputs = tokenizer(prompt, return_tensors="pt")
    generate_ids = model.generate(inputs.input_ids, max_length=30)
    output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(output)


def test_vision():
    from transformers import AutoTokenizer, OPTForCausalLM, OPTModel
    from PIL import Image
    import requests
    model_name_or_path = '/home/yuhaiyang/zlw/hisdoc/data/models--facebook--opt-125m'

    model = ClipOPTForCausalLM.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)   
    multimodal_cfg = {
        'img_patch_token': '<imgpatch>',
        'img_start_token': '<img>',
        'img_end_token': '</img>',
        'img_token_len': 256
    }
    model.initialize_vision_tokenizer(tokenizer, multimodal_cfg)
    img_processor = model.get_model().get_img_processor()['img_processor']


    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    imgs = img_processor(images=image, return_tensors="pt")
    print(imgs.keys())
    print(imgs['pixel_values'].shape)


    multimodal_cfg = {
        'img_patch_token': '<imgpatch>',
        'img_start_token': '<img>',
        'img_end_token': '</img>',
        'img_token_len': 256
    }
    # model.initialize_vision_tokenizer(tokenizer, multimodal_cfg)

    prompt = '<img>' + '<imgpatch>'*256 + '</img>' + 'Hello, Who are you?'
    inputs = tokenizer(prompt, return_tensors="pt")

    res = model(**inputs, images=imgs)

    # generate_ids = model.generate(inputs.input_ids, max_length=30)
    # output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(res)
    pass

def test_multimodal():
    from transformers import AutoTokenizer, OPTForCausalLM, OPTModel
    from PIL import Image
    import requests
    model_name_or_path = '/home/yuhaiyang/zlw/hisdoc/data/models--facebook--opt-125m'

    model = ClipOPTForCausalLM.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)   
    multimodal_cfg = {
        'img_patch_token': '<imgpatch>',
        'img_start_token': '<img>',
        'img_end_token': '</img>',
        'img_token_len': 256
    }
    model.initialize_vision_tokenizer(tokenizer, multimodal_cfg)
    img_processor = model.get_model().get_img_processor()['img_processor']


    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    imgs = img_processor(images=image, return_tensors="pt")
    print(imgs.keys())
    print(imgs['pixel_values'].shape)


    multimodal_cfg = {
        'img_patch_token': '<imgpatch>',
        'img_start_token': '<img>',
        'img_end_token': '</img>',
        'img_token_len': 256
    }
    # model.initialize_vision_tokenizer(tokenizer, multimodal_cfg)

    prompt = '<img>' + '<imgpatch>'*256 + '</img>' + 'Hello, Who are you?'
    inputs = tokenizer(prompt, return_tensors="pt")

    res = model(**inputs, images=imgs)

    # generate_ids = model.generate(inputs.input_ids, max_length=30)
    # output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(res)
    pass

if __name__ == '__main__':
    
    test_multimodal()