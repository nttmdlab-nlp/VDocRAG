from typing import List, Tuple

from datasets import load_dataset
from torch.utils.data import Dataset
from PIL import Image
import os
import json
from vdocrag.vdocretriever.arguments import DataArguments

import logging
logger = logging.getLogger(__name__)


class TrainDataset(Dataset):
    def __init__(self, data_args: DataArguments, trainer = None):
        self.data_args = data_args
        self.train_data = load_dataset(
            self.data_args.dataset_name,
            self.data_args.dataset_config,
            data_files=self.data_args.dataset_path,
            split=self.data_args.dataset_split,
            cache_dir=self.data_args.dataset_cache_dir,
        )
        if not self.data_args.pretrain:
            self.corpus  = load_dataset(
                self.data_args.corpus_name,
                self.data_args.corpus_config,
                data_files=self.data_args.corpus_path,
                split=self.data_args.corpus_split,
                cache_dir=self.data_args.dataset_cache_dir,
            )

            self.docid2idx = {}
            if 'doc_id' in self.corpus.features:
                for idx, docid in enumerate(self.corpus['doc_id']):
                    self.docid2idx[str(docid)] = idx
            else:
                for idx in range(len(self.corpus)):
                    self.docid2idx[str(idx)] = idx

        self.trainer = trainer

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item) -> Tuple[str, List[str]]:
        group = self.train_data[item]
        epoch = int(self.trainer.state.epoch)

        _hashed_seed = hash(item + self.trainer.args.seed)

        query = group['query']
        if self.data_args.pretrain:
            image = group['image']
        else:
            relevant_docids = group['relevant_doc_ids']

            if self.data_args.positive_document_no_shuffle:
                docid = relevant_docids[0]
            else:
                docid = relevant_docids[(_hashed_seed + epoch) % len(relevant_docids)]

            image = image = self.corpus[self.docid2idx[docid]]['image']

        return query, image


class EncodeDataset(Dataset):
    def __init__(self, data_args: DataArguments):
        self.data_args = data_args
        if self.data_args.encode_is_query:
            self.encode_data = load_dataset(
                self.data_args.dataset_name,
                self.data_args.dataset_config,
                data_files=self.data_args.dataset_path,
                split=self.data_args.dataset_split,
                cache_dir=self.data_args.dataset_cache_dir,
            )
        else:    
            self.encode_data = load_dataset(
                self.data_args.corpus_name,
                self.data_args.corpus_config,
                data_files=self.data_args.corpus_path,
                split=self.data_args.corpus_split,
                cache_dir=self.data_args.dataset_cache_dir,
            )

        if self.data_args.dataset_number_of_shards > 1:
            self.encode_data = self.encode_data.shard(
                num_shards=self.data_args.dataset_number_of_shards,
                index=self.data_args.dataset_shard_index,
            )
        
    def __len__(self):
        return len(self.encode_data)

    def __getitem__(self, item) -> Tuple[str, str]:
        data = self.encode_data[item]
        text, image = None, None
        if self.data_args.encode_is_query:
            id = data['query_id']
            text = data['query']
        else:
            id = data['doc_id']
            image = data['image']
        return id, text, image
