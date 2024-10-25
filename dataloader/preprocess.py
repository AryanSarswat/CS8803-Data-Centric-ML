import pickle

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm
from transformers import DistilBertConfig, DistilBertModel, DistilBertTokenizer

if __name__ == "__main__":
    meta_data_file = "../LLaVA-CC3M-Pretrain-595K/metadata.json"
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

    with open('../LLaVA-CC3M-Pretrain-595K/preprocessed_image_text_pairs.pkl', 'wb') as f:
        pickle.dump(image_text_pairs, f)
    
    

