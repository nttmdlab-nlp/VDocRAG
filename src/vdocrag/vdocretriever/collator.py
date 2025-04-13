import torch
import logging
from typing import List, Tuple
from dataclasses import dataclass
from transformers import PreTrainedTokenizer, ProcessorMixin
from vdocrag.vdocretriever.arguments import DataArguments
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class TrainCollator:
    data_args: DataArguments
    tokenizer: PreTrainedTokenizer
    processor: ProcessorMixin

    def build_image_attention_mask(self, seq_len, input_lengths):
        image_attention_masks = []
        for input_len in input_lengths:
            image_attention_mask = torch.tril(torch.ones(seq_len, seq_len), diagonal=0)
            image_attention_mask[input_len:, :input_len-1] = 0 
            image_attention_masks.append(image_attention_mask.unsqueeze(0))
        image_attention_masks = torch.cat(image_attention_masks, dim=0)
        return image_attention_masks

    def __call__(self, features: List[Tuple[str, List[str]]]):
        all_queries = [f[0] for f in features]
        all_images = [f[-1] for f in features]

        q_collated = self.tokenizer(
            all_queries,
            padding=False, 
            truncation=True,
            max_length=self.data_args.query_max_len-1 if self.data_args.append_eos_token else self.data_args.query_max_len,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=True,
        )

        d_collated = {}
        collated_list = [self.processor("<|image_1|>\nWhat is shown in this image?", image, return_tensors="pt") for image in all_images]        
        d_collated['input_ids'] = [d['input_ids'][0].tolist() for d in collated_list]

        if self.data_args.append_eos_token:
            q_collated['input_ids'] = [q + [self.tokenizer.eos_token_id] for q in q_collated['input_ids']]
            d_collated['input_ids'] = [d + [self.tokenizer.eos_token_id] for d in d_collated['input_ids']]

        if self.data_args.pretrain:
            p_collated = {}
            all_input_ids, all_label_ids, input_lengths = [], [], []

            for i, ocr in enumerate(all_queries):
                prompt_input_ids = torch.tensor(d_collated['input_ids'][i]).unsqueeze(0)
                answer = f'{ocr}<|end|>\n<|endoftext|>'
                answer_input_ids = self.tokenizer(
                    answer, add_special_tokens=False, max_length=self.data_args.answer_max_len, truncation=True, return_tensors='pt')['input_ids']
                input_ids = torch.cat([prompt_input_ids, answer_input_ids], dim=1)
                labels = torch.cat(
                    [
                        torch.tensor([-100] * len(prompt_input_ids[0])).unsqueeze(0),
                        answer_input_ids,
                    ],
                    dim=1,
                )
                all_input_ids.append(input_ids.squeeze(0).unsqueeze(1))
                all_label_ids.append(labels.squeeze(0).unsqueeze(1))
                input_lengths.append(len(prompt_input_ids[0]))

            input_ids = torch._C._nn.pad_sequence(
                all_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            ).squeeze(2)
            labels = torch._C._nn.pad_sequence(
                all_label_ids, batch_first=True, padding_value=-100
            ).squeeze(2)

            p_collated['input_ids'] = input_ids
            p_collated['labels'] = labels

            if self.data_args.image_attention_mask:
                image_attention_mask = self.build_image_attention_mask(input_ids.size()[1], input_lengths)
                p_collated['attention_mask'] = image_attention_mask.unsqueeze(1)
        else:
            p_collated = None

        q_collated = self.tokenizer.pad(
            q_collated,
            padding=True, 
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors='pt',
        )
        d_collated = self.tokenizer.pad(
            d_collated,
            padding=True, 
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors='pt',
        )

        d_collated['pixel_values'] = torch.stack([d['pixel_values'][0] for d in collated_list], dim=0)
        d_collated['image_sizes'] = torch.stack([d['image_sizes'][0] for d in collated_list], dim=0)
        if self.data_args.pretrain:
            p_collated['pixel_values'] = d_collated['pixel_values']
            p_collated['image_sizes'] = d_collated['image_sizes']

        return q_collated, d_collated, p_collated

@dataclass
class EncodeCollator:
    data_args: DataArguments
    tokenizer: PreTrainedTokenizer
    processor: ProcessorMixin

    def __call__(self, features: List[Tuple[str, str]]):
        text_ids = [x[0] for x in features]
        texts = [x[1] for x in features]
        images = [x[-1] for x in features]

        if self.data_args.encode_is_query:
            collated = self.tokenizer(
                texts,
                padding=False, 
                truncation=True,
                max_length=self.data_args.query_max_len-1 if self.data_args.append_eos_token else self.data_args.query_max_len,
                return_attention_mask=False,
                return_token_type_ids=False,
                add_special_tokens=True,
            )
        else:
            collated = {}
            collated_list = [self.processor("<|image_1|>\nWhat is shown in this image?", image, return_tensors="pt") for image in images]
            collated['input_ids'] = [d['input_ids'][0].tolist() for d in collated_list]

        if self.data_args.append_eos_token:
            collated['input_ids'] = [x + [self.tokenizer.eos_token_id] for x in collated['input_ids']]
            
        collated = self.tokenizer.pad(
            collated,
            padding=True, 
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors='pt',
        )
        if not self.data_args.encode_is_query:
            collated['pixel_values'] = torch.stack([d['pixel_values'][0] for d in collated_list], dim=0)
            collated['image_sizes'] = torch.stack([d['image_sizes'][0] for d in collated_list], dim=0)
        
        return text_ids, collated