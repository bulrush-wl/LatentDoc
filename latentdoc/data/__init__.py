
import torch
import transformers
from dataclasses import dataclass, field
# from latentdoc.data.conversation_dataset import ConversationDataset
from latentdoc.data.simple_conversation_dataset import SimpleConversationDateset

IGNORE_INDEX = -100

@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):

        images = [instance['images'] for instance in instances]  
        images = torch.stack(images, dim=0)   # b, c, h, w

        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)   # b, l
            
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX)                 # b, l
        
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            images=images,
        )

        return batch


def make_supervised_data_module(tokenizer, img_processor, data_args, multimodal_cfg):


    train_dataset = SimpleConversationDateset(
        datasets = data_args.datasets,
        tokenizer = tokenizer,
        img_processor = img_processor,
        multimodal_cfg = multimodal_cfg,
    )

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

