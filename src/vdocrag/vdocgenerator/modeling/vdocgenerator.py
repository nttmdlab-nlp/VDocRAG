from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.distributed as dist
from torch import nn, Tensor

from transformers import PreTrainedModel, AutoModel, AutoModelForCausalLM 
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

from transformers.file_utils import ModelOutput
from vdocrag.vdocgenerator.arguments import ModelArguments, VDocGeneratorTrainingArguments as TrainingArguments

import logging
logger = logging.getLogger(__name__)


@dataclass
class DecoderOutput(ModelOutput):
    loss: Optional[Tensor] = None


class VDocGenerator(nn.Module):
    TRANSFORMER_CLS = AutoModelForCausalLM 

    def __init__(self,
                 decoder: PreTrainedModel,
                 ):
        super().__init__()
        self.config = decoder.config
        self.decoder = decoder
        self.is_ddp = dist.is_initialized()
        if self.is_ddp:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def forward(self, inputs: Dict[str, Tensor] = None, use_cache: bool = True):
        outputs = self.decode(inputs, use_cache=use_cache)

        # for training
        if self.training:
            loss = outputs.loss

            if self.is_ddp:
                loss = loss * self.world_size  # counter average weight reduction

        # for eval
        else:
            loss = None
        return DecoderOutput(
            loss=loss,
        )

    def gradient_checkpointing_enable(self, **kwargs):
        self.decoder.model.gradient_checkpointing_enable()

    @classmethod
    def build(
            cls,
            model_args: ModelArguments,
            train_args: TrainingArguments,
            **hf_kwargs,
    ):  
        base_model = cls.TRANSFORMER_CLS.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0
        if model_args.lora or model_args.lora_name_or_path:
            if train_args.gradient_checkpointing:
                base_model.enable_input_require_grads()
            if model_args.lora_name_or_path:
                lora_config = LoraConfig.from_pretrained(model_args.lora_name_or_path, **hf_kwargs)
                lora_model = PeftModel.from_pretrained(base_model, model_args.lora_name_or_path, is_trainable=True)
            else:
                lora_config = LoraConfig(
                    base_model_name_or_path=model_args.model_name_or_path,
                    task_type=TaskType.FEATURE_EXTRACTION,
                    r=model_args.lora_r,
                    lora_alpha=model_args.lora_alpha,
                    lora_dropout=model_args.lora_dropout,
                    target_modules=model_args.lora_target_modules.split(','),
                    inference_mode=False
                )
                lora_model = get_peft_model(base_model, lora_config)
            model = cls(
                decoder=lora_model,
            )
        else:
            model = cls(
                decoder=base_model,
            )
        return model

    @classmethod
    def load(cls,
             model_name_or_path: str,
             lora_name_or_path: str = None,
             **hf_kwargs):
        base_model = cls.TRANSFORMER_CLS.from_pretrained(model_name_or_path, **hf_kwargs)
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0
        if lora_name_or_path:
            lora_config = LoraConfig.from_pretrained(lora_name_or_path, **hf_kwargs)
            lora_model = PeftModel.from_pretrained(base_model, lora_name_or_path, config=lora_config)
            lora_model = lora_model.merge_and_unload()
            model = cls(
                decoder=lora_model,
            )
        else:
            model = cls(
                decoder=base_model,
            )
        return model

    def save(self, output_dir: str):
        self.decoder.save_pretrained(output_dir)

    def decode(self, input, use_cache=True):
        return self.decoder(**input, use_cache=use_cache)

    def generate(self, input, generation_args, use_cache=True):
        return self.decoder.generate(**input, **generation_args, use_cache=use_cache)