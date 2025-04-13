import json
from argparse import ArgumentParser
from datasets import load_dataset

parser = ArgumentParser()
parser.add_argument('--dataset_name', type=str, required=True)
parser.add_argument('--dataset_config', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
args = parser.parse_args()

data = load_dataset(
    args.dataset_name,
    args.dataset_config,
    split="test",
)

with open(args.output, 'w') as f_out:
    for d in data:
        query_id = d['query_id']
        for docid in d['relevant_doc_ids']:
            f_out.write(f'{query_id} Q0 {docid} 1\n')
