import numpy as np
import torch
from torch import nn
from transformers import DistilBertConfig, DistilBertModel, DistilBertTokenizer
import pandas as pd
from tqdm import tqdm
import pickle
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='../LLaVA-CC3M-Pretrain-595K')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    meta_data_file = os.path.join(args.dataset, "metadata.json")
    annotations = pd.read_json(meta_data_file)

    image_text_pairs = []
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    for index, row in tqdm(annotations.iterrows(), total=len(annotations)):
        tokenized_caption = tokenizer(row['caption'], padding="max_length", truncation=True, max_length=50, return_tensors="pt")
        input_ids = tokenized_caption['input_ids']
        attention_mask = tokenized_caption['attention_mask']
        image_text_pairs.append({
            'image_path' : row['image'],
            'input_ids' : input_ids,
            'attention_mask' : attention_mask
        })

    with open(os.path.join(args.dataset, 'preprocessed_image_text_pairs.pkl'), 'wb') as f:
        pickle.dump(image_text_pairs, f)
    
    

