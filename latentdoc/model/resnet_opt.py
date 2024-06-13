from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from latentdoc.model.llm.opt import build_opt_causal_lm
from latentdoc.model.vision_encoder.resnet import build_resnet152_and_img_processor

class Rsenet_OPT(nn.Module):

    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()

        self.config = config

        self.img_processor, self.vision_encoder = build_resnet152_and_img_processor()

        self.tokenizer, self.llm = build_opt_causal_lm(config)

        self.mm_projector = nn.Linear(2048, self.llm.config.hidden_size)

        self._init_multimodal_module()
        self._resize_embedding()
        self._init_mm_projector()

    def _init_mm_projector(self, ):

        std = self.llm.config.init_std
        self.mm_projector.weight.data.normal_(mean=0.0, std=std)
        if self.mm_projector.bias is not None:
            self.mm_projector.bias.data.zero_()
        
    def _resize_embedding(self, ):

        raw_llm_vocab = self.llm.config.vocab_size

        if len(self.tokenizer) < raw_llm_vocab:
            print(f'The raw embedding size is {raw_llm_vocab}, the current number of token is {len(self.tokenizer)}, no need to do the resize for embedding.')

        else:
            self.llm.resize_token_embeddings(len(self.tokenizer))  # do not know

            # resize_token_embeddings will do the following things, too
            input_embeddings = self.llm.get_input_embeddings().weight.data
            output_embeddings = self.llm.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-self.num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-self.num_new_tokens].mean(dim=0, keepdim=True)

            input_embeddings[-self.num_new_tokens:] = input_embeddings_avg
            output_embeddings[-self.num_new_tokens:] = output_embeddings_avg

            print(f'\nAdded {self.num_new_tokens} special tokens, the vocab size of llm from {raw_llm_vocab} to {self.llm.config.vocab_size} \n')

        pass

    def _init_multimodal_module(self, ):

        self.img_token_len = self.config.img_token_len

        self.special_tokens = self.config.special_tokens

        self.num_new_tokens = self.tokenizer.add_special_tokens({
        'additional_special_tokens': list(self.special_tokens.values())
        })

        self.img_patch_token_id = self.tokenizer.convert_tokens_to_ids(self.special_tokens.img_patch_token)

        self.im_start_token_id = self.tokenizer.convert_tokens_to_ids(self.special_tokens.im_start_token)
        self.im_end_token_id = self.tokenizer.convert_tokens_to_ids(self.special_tokens.im_end_token)

        self.img_start_token_id = self.tokenizer.convert_tokens_to_ids(self.special_tokens.img_start_token)
        self.img_end_token_id = self.tokenizer.convert_tokens_to_ids(self.special_tokens.img_end_token)

    def get_tokenizer(self,):
        return self.tokenizer

    def get_img_processor(self, ):
        return self.img_processor

    def embed_tokens(self, input_ids):
        return self.llm.get_input_embeddings()(input_ids)

    def embed_images(self, images):

        img_features = self.vision_encoder(images)  # resnet  b,c,h,w
        
        img_features = img_features.flatten(2).permute(0, 2, 1)
        img_features = self.mm_projector(img_features)

        return img_features

    def multimodal_process(self, input_ids, input_embeddings, img_features):


        dummy_image_features = torch.zeros(256, 2048, device=input_embeddings.device, dtype=input_embeddings.dtype)
        dummy_image_features = self.mm_projector(dummy_image_features)

        new_input_embeds = []
        for cur_input_ids, cur_input_embeds, cur_image_features in zip(input_ids, input_embeddings, img_features):

            # multimodal LLM, but the current sample is not multimodal
            if (cur_input_ids == self.img_patch_token_id).sum() == 0:
                cur_input_embeds = cur_input_embeds + (0. * dummy_image_features).sum()
                new_input_embeds.append(cur_input_embeds)
                continue

            # check the input
            if (cur_input_ids == self.im_start_token_id).sum() != (cur_input_ids == self.im_end_token_id).sum() :
                    raise ValueError("The number of input message start tokens and input message end tokens should be the same.")
            if (cur_input_ids == self.img_start_token_id).sum() != (cur_input_ids == self.img_end_token_id).sum() :
                    raise ValueError("The number of image start tokens and image end tokens should be the same.")


            image_start_tokens = torch.where(cur_input_ids == self.img_start_token_id)[0]

            # add the img features to the input embedding, only excute once!
            for image_start_token_pos in image_start_tokens:
                num_patches = cur_image_features.shape[0]

                # check the input
                if cur_input_ids[image_start_token_pos + num_patches + 1] != self.img_end_token_id:
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
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        img_features = self.embed_images(images)
        input_embeddings = self.embed_tokens(input_ids)
        multimodal_input_embeddings = self.multimodal_process(input_ids, input_embeddings, img_features)


        return self.llm(
            input_ids=None, inputs_embeds=multimodal_input_embeddings, attention_mask=attention_mask, labels=labels,
            past_key_values=past_key_values, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )


        # hidden_states = outputs[0]
        # logits = self.lm_head(hidden_states).contiguous()

        # # logits

        # loss = None
        # if labels is not None:
        #     # move labels to correct device to enable model parallelism
        #     labels = labels.to(logits.device)
        #     # Shift so that tokens < n predict n
        #     shift_logits = logits[..., :-1, :].contiguous()
        #     shift_labels = labels[..., 1:].contiguous()
        #     # Flatten the tokens
        #     loss_fct = CrossEntropyLoss()
        #     loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        # if not return_dict:
        #     output = (logits,) + outputs[1:]
        #     return (loss,) + output if loss is not None else output

        # return CausalLMOutputWithPast(
        #     loss=loss,
        #     logits=logits,
        #     past_key_values=outputs.past_key_values,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )


    def generate(
        self, input_ids, images, inputs_embeds=None, **kwargs
    ):
        img_features = self.embed_images(images)
        input_embeddings = self.embed_tokens(input_ids)
        multimodal_input_embeddings = self.multimodal_process(input_ids, input_embeddings, img_features)

        return self.llm.generate(input_ids=input_ids, inputs_embeds=multimodal_input_embeddings, **kwargs)

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

    multimodal_cfg = {
        'img_token_len': 256,
        'model_max_length': 4096,
        'output_attentions': True,
        'output_hidden_states': True,
        'return_dict': True,
        'special_tokens': special_tokens
    }

    multimodal_cfg = edict(multimodal_cfg)

    model = Rsenet_OPT(multimodal_cfg)

    img_processor, tokenizer = model.get_img_processor(), model.get_tokenizer()

    ds = SimpleConversationDateset(datasets='zhongtie_doc', img_processor=img_processor, tokenizer=tokenizer, multimodal_cfg=multimodal_cfg)


    data = ds[0]
    data['input_ids'] = data['input_ids'].unsqueeze(dim=0)
    data['labels'] = data['labels'].unsqueeze(dim=0)
    data['images'] = data['images'].unsqueeze(dim=0)
    data['input_ids'] = torch.tensor(torch.arange(1,3000), dtype=torch.long).unsqueeze(dim=0)
    data['labels'] = torch.tensor(torch.arange(1,3000), dtype=torch.long).unsqueeze(dim=0)

    output = model(**data)
    print(output.loss)
    pass

if __name__ == '__main__':
    test()