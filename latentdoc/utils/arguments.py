from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import transformers


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    vision_encoder: Optional[str] = field(default="")
    ae: Optional[str] = field(default="")
    img_size: int = field(default=512)
    img_token_len: int = field(default=256)
    use_cache: bool = field(default=False)
    freeze_vision_encoder: bool = field(default=False)
    freeze_lm_model: bool = field(default=False)
    freeze_ae: bool = field(default=True)
   

@dataclass
class DataArguments:
    datasets: str = field(default=None, metadata={"help": "combinations of the training data."})
    # multimodal_cfg: str = field(default=None, metadata={"help": "the multimodal_config path"})
    # image_token_len: int = 256
    # image_aspect_ratio: str = 'square'
    # conversation_version: str = 'mpt'
    # conversation_version: str = 'v0'
    # conversation_version: str = 'v1'
    # conversation_version: str = 'opt'
    # box_limit: int = 0


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    resume: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    force_fsdp: bool = field(default=False)
    interleave: bool = field(default=False)
    with_box: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    lora_enable: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"