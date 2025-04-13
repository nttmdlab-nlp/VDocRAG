import logging
import os
import pickle
import sys
import json
from contextlib import nullcontext

import numpy as np
from tqdm import tqdm

import torch
import time
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoProcessor
from transformers import (
    HfArgumentParser,
)

from vdocrag.vdocgenerator.arguments import ModelArguments, DataArguments, \
    VDocGeneratorTrainingArguments as TrainingArguments
from vdocrag.vdocgenerator.dataset import DecodeDataset
from vdocrag.vdocgenerator.collator import DecodeCollator
from vdocrag.vdocgenerator.modeling import DecoderOutput, VDocGenerator

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    if training_args.local_rank > 0 or training_args.n_gpu > 1:
        raise NotImplementedError('Multi-GPU encoding is not supported.')

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    processor = AutoProcessor.from_pretrained(model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
                                              cache_dir=model_args.cache_dir,
                                              trust_remote_code=True,)

    tokenizer = processor.tokenizer

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'right'

    if training_args.bf16:
        torch_dtype = torch.bfloat16
    elif training_args.fp16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    
    model = VDocGenerator.load(
        model_args.model_name_or_path,
        normalize=model_args.normalize,
        lora_name_or_path=model_args.lora_name_or_path,
        trust_remote_code=True,
        cache_dir=model_args.cache_dir,
        torch_dtype=torch_dtype,
        _attn_implementation='flash_attention_2',
    )

    decode_dataset = DecodeDataset(
        data_args=data_args,
    )

    decode_collator = DecodeCollator(
        data_args=data_args,
        tokenizer=tokenizer,
        processor=processor,
    )

    decode_loader = DataLoader(
        decode_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=decode_collator,
        shuffle=False,
        drop_last=False,
        num_workers=training_args.dataloader_num_workers,
    )
    responses = {}
    model = model.to(training_args.device)
    model.eval()

    generation_args = { 
        "max_new_tokens": 64, 
        "temperature": 0.0, 
        "do_sample": False, 
        "eos_token_id": tokenizer.eos_token_id,
    } 

    # TODO batch > 1
    for (batch_ids, answers, batch) in tqdm(decode_loader):
        with nullcontext():
            with torch.no_grad():
                for k, v in batch.items():
                    batch[k] = v.to(training_args.device)
                generate_ids = model.generate(batch, 
                                              generation_args=generation_args,                                              
                                              )
                generate_ids = generate_ids[:, batch['input_ids'].shape[1]:]
                response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                response = response.strip()
                responses[batch_ids[0]] = {"ground_truth": answers[0], "prediction": response}

    if not os.path.exists(os.path.dirname(data_args.output_path)):
        os.makedirs(os.path.dirname(data_args.output_path))
    with open(data_args.output_path, 'w') as f:
        json.dump(responses, f)

if __name__ == "__main__":
    main()
