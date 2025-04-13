import torch
import logging
from typing import List, Tuple
from dataclasses import dataclass
from transformers import PreTrainedTokenizer, ProcessorMixin
from vdocrag.vdocgenerator.arguments import DataArguments, TrainingArguments
from transformers.feature_extraction_utils import BatchFeature
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class TrainCollator:
    data_args: DataArguments
    tokenizer: PreTrainedTokenizer
    processor: ProcessorMixin

    def __call__(self, features: List[Tuple[str, List[str]]]):
        """
        Collate function for training.
        :param features: list of (query, documents) tuples
        :return: tokenized query_ids, document_ids
        """

        all_queries = [f[0] for f in features]
        all_answers = [f[1] for f in features]
        all_images = [f[2] for f in features]

        collated = {}
        all_input_ids, all_label_ids, pixel_values, image_sizes = [], [], [], []
        for i, (query, answer, images) in enumerate(zip(all_queries, all_answers, all_images)):
            image_tokens = "\n".join([f"<|image_{i+1}|>" for i in range(len(images))])
            messages = [{"role": "user", "content": f"{image_tokens}\n{query}"}]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) 
            processed = self.processor(prompt, images, return_tensors="pt")
            answer = f'{answer}<|end|>\n<|endoftext|>'
            answer_input_ids = self.tokenizer(
                answer, add_special_tokens=False, return_tensors='pt'
            )['input_ids']
            prompt_input_ids = processed['input_ids']
            input_ids = torch.cat([prompt_input_ids, answer_input_ids], dim=1)
            labels = torch.cat(
                [
                    torch.tensor([-100] * len(prompt_input_ids[0])).unsqueeze(0),
                    answer_input_ids,
                ],
                dim=1,
            )
            # prepare expected shape for pad_sequence
            all_input_ids.append(input_ids.squeeze(0).unsqueeze(1))
            all_label_ids.append(labels.squeeze(0).unsqueeze(1))
            pixel_values.append(processed['pixel_values'])
            image_sizes.append(processed['image_sizes'])

        input_ids = torch._C._nn.pad_sequence(
            all_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        ).squeeze(2)
        labels = torch._C._nn.pad_sequence(
            all_label_ids, batch_first=True, padding_value=-100
        ).squeeze(2)

        collated['input_ids'] = input_ids
        collated['labels'] = labels
        collated['pixel_values'] = torch.cat(pixel_values, dim=0)
        collated['image_sizes'] = torch.cat(image_sizes, dim=0)
        collated['attention_mask'] = collated['input_ids'].ne(self.tokenizer.pad_token_id)

        return collated

@dataclass
class DecodeCollator:
    data_args: DataArguments
    tokenizer: PreTrainedTokenizer
    processor: ProcessorMixin

    def __call__(self, features: List[Tuple[str, str]]):
        """
        Collate function for encoding.
        :param features: list of (id, text) tuples
        """
        query_ids = [f[0] for f in features]
        all_queries = [f[1] for f in features]
        all_answers = [f[2] for f in features]
        all_images = [f[3] for f in features]

        collated = defaultdict(list)
        pixel_values, image_sizes = [], []
        for i, (query, images) in enumerate(zip(all_queries, all_images)):
            image_tokens = "\n".join([f"<|image_{i+1}|>" for i in range(len(images))])
            messages = [{"role": "user", "content": f"{image_tokens}\n{query}"}]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) 
            processed = self.processor(prompt, images, return_tensors="pt")
            prompt_input_ids = processed['input_ids']
            collated['input_ids'].append(prompt_input_ids)
            pixel_values.append(processed['pixel_values'])
            image_sizes.append(processed['image_sizes'])

        collated['input_ids'] = torch.cat(collated['input_ids'], dim=0)
        collated['pixel_values'] = torch.cat(pixel_values, dim=0)
        collated['image_sizes'] = torch.cat(image_sizes, dim=0)
        
        return query_ids, all_answers, collated