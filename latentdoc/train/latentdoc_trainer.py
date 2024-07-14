import os
import torch
import torch.nn as nn
import math
from transformers import Trainer
from transformers.trainer_pt_utils import get_parameter_names
from transformers import get_scheduler
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from typing import Dict, Optional, Sequence

from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineRestartLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, restart_epochs, min_lr=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.restart_epochs = restart_epochs
        self.min_lr = min_lr
        super(WarmupCosineRestartLR, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        current_epoch = self.last_epoch % self.restart_epochs
        if current_epoch < self.warmup_epochs:
            # Warmup phase
            warmup_factor = float(current_epoch) / float(max(1, self.warmup_epochs))
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing phase
            cosine_factor = 0.5 * (1 + math.cos(math.pi * (current_epoch - self.warmup_epochs) / (self.restart_epochs - self.warmup_epochs)))
            return [self.min_lr + (base_lr - self.min_lr) * cosine_factor for base_lr in self.base_lrs]


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        
        return unwrap_model(model.module)

    else:
        return model


class LatentDocTrainer(Trainer):

    def _safe_save(self, output_dir: str):
        """Collects the state dict and dump to disk."""
        state_dict = self.model.state_dict()
        if self.args.should_save:
            cpu_state_dict = {
                key: value.cpu()
                for key, value in state_dict.items()
            }
            del state_dict
            self._save(output_dir, state_dict=cpu_state_dict)  # noqa

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            # Save the model
            _state_dict = state_dict
            if _state_dict is None:
                # Only save the model itself if we are using distributed training
                model_to_save = unwrap_model(self.model)
                _state_dict = model_to_save.state_dict()

            weight_to_save = {}
            keys_to_match = ['mm_projector', 'embed_tokens', 'embed_in']
            for k, v in _state_dict.items():
                if any(key_match in k for key_match in keys_to_match):
                    weight_to_save[k] = v

            current_folder = output_dir.split('/')[-1]
            parent_folder = os.path.dirname(output_dir)
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))

        super(LatentDocTrainer, self)._save(output_dir, state_dict)

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if 'vision_encoder' in n and n in decay_parameters and p.requires_grad
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if 'vision_encoder' in n and n not in decay_parameters and p.requires_grad],
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if 'vision_encoder' not in n and n in decay_parameters and p.requires_grad],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if 'vision_encoder' not in n and n not in decay_parameters and p.requires_grad
                    ],
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate,
                },
            ]
            for idx, group in enumerate(optimizer_grouped_parameters):
                print(idx, len(group['params']), group['lr'])
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.

        """
        if self.lr_scheduler is None:
            # self.args.lr_scheduler_kwargs = {'num_cycles': self.args.num_cycles}
            if self.args.lr_scheduler_type == 'cosine_with_restarts':
                warmup_steps = int(self.args.warmup_ratio * num_training_steps/ self.args.num_train_epochs)
                # warmup_steps = int(self.args.warmup_ratio * num_training_steps)
                restart_steps = int(num_training_steps / self.args.num_train_epochs)
                self.lr_scheduler = WarmupCosineRestartLR(self.optimizer, warmup_steps, num_training_steps, restart_steps)

            else:
                self.lr_scheduler = get_scheduler(
                    self.args.lr_scheduler_type,
                    optimizer=self.optimizer if optimizer is None else optimizer,
                    num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                    num_training_steps=num_training_steps,
                )

        self._created_lr_scheduler = True
        return self.lr_scheduler
        