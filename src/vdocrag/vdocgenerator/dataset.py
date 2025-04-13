import random
from typing import List, Tuple

from datasets import load_dataset
from torch.utils.data import Dataset
from PIL import Image
import os
import json
from vdocrag.vdocgenerator.arguments import DataArguments
from scipy.special import softmax
from collections import defaultdict
import numpy as np

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

        self.corpus = load_dataset(
            self.data_args.corpus_name,
            self.data_args.corpus_config,
            data_files=self.data_args.corpus_path,
            split=self.data_args.corpus_split,
            cache_dir=self.data_args.dataset_cache_dir,
        )

        self.docid2idx = {}
        for idx, doc_id in enumerate(self.corpus['doc_id']):
            self.docid2idx[str(doc_id)] = idx

        self.retrieved_docs = defaultdict(list)
        with open(self.data_args.retrieval_results_path) as f:
            lines = f.read().splitlines()
            for line in lines:
                query_id, doc_id, score = line.split()
                self.retrieved_docs[query_id].append(doc_id)
            self.retrieved_docs[query_id] = self.retrieved_docs[query_id][:self.data_args.top_k]

        self.trainer = trainer

    def __len__(self):
        return len(self.train_data)

    def _get_image(self, doc_id):
        image = self.corpus[self.docid2idx[doc_id]]['image']
        return image

    def __getitem__(self, item) -> Tuple[str, List[str]]:
        group = self.train_data[item]
        query_id = group['query_id']
        query = group['query']
        answer = group['answers'][0]
        images = [self._get_image(doc_id) for doc_id in self.retrieved_docs[query_id]]

        return query, answer, images


class DecodeDataset(Dataset):

    def __init__(self, data_args: DataArguments):
        self.data_args = data_args
        self.test_data = load_dataset(
            self.data_args.dataset_name,
            self.data_args.dataset_config,
            data_files=self.data_args.dataset_path,
            split=self.data_args.dataset_split,
            cache_dir=self.data_args.dataset_cache_dir,
        )

        self.corpus = load_dataset(
            self.data_args.corpus_name,
            self.data_args.corpus_config,
            data_files=self.data_args.corpus_path,
            split=self.data_args.corpus_split,
            cache_dir=self.data_args.dataset_cache_dir,
        )

        self.docid2idx = {}
        for idx, doc_id in enumerate(self.corpus['doc_id']):
            self.docid2idx[str(doc_id)] = idx

        self.retrieved_docs = defaultdict(list)
        with open(self.data_args.retrieval_results_path) as f:
            lines = f.read().splitlines()
            for line in lines:
                query_id, doc_id, score = line.split()
                self.retrieved_docs[query_id].append(doc_id)

        for i, d in enumerate(self.test_data):
            candidates = []
            for image in self.retrieved_docs[query_id][:self.data_args.top_k]:
                candidates.append(image)
            self.test_data[i]["candidates"] = candidates
        
    def __len__(self):
        return len(self.test_data)

    def _get_image(self, doc_id):
        image = self.corpus[self.docid2idx[doc_id]]['image']
        return image

    def __getitem__(self, item) -> Tuple[str, str]:
        data = self.test_data[item]
        query_id = data['query_id']
        query = data["query"]
        answers = data['answers']
        images = [self._get_image(doc_id) for doc_id in data["candidates"]]

        return query_id, query, answers, images
