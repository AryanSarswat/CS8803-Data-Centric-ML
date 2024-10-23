import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import ToTensor
import pickle
from transformers import SiglipImageProcessor, AutoProcessor
from PIL import Image

class CC3MDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        if not root_dir.endswith('/'):
            root_dir = root_dir + '/'
        annotations = pd.read_json(root_dir+"metadata.json")
        self.image_paths = root_dir + 'images/' + annotations['image']
        self.captions = annotations['caption']
        self.processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
        self.transform = transform

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        pil_image = Image.open(image_path)
        caption = self.captions[index]
        inputs = self.processor(text=caption , images=pil_image, padding="max_length", return_tensors="pt")
        image = inputs['pixel_values'][0]
        input_ids = inputs['input_ids'][0]
        return image, input_ids

# class CC3MDataset(Dataset):
#     def __init__(self, pickle_file, root_dir, transform=None):
#         with open(pickle_file, 'rb') as f:
#             self.annotations = pickle.load(f)
#         self.root_dir = root_dir
#         self.transform = transform

#     def __len__(self):
#         return len(self.annotations)

#     def __getitem__(self, index):
#         img_path = os.path.join(self.root_dir, self.annotations[index]['image_path'])
#         image = read_image(img_path).float()
#         input_ids = self.annotations[index]['input_ids']
#         attention_mask = self.annotations[index]['attention_mask']

#         if self.transform:
#             image = self.transform(image)

#         return (image, (input_ids, attention_mask))

#     def show_image(self, index):
#         img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
#         image = read_image(img_path)
#         image = image.permute(1, 2, 0)
#         label = [self.annotations.iloc[index, 1]]
        
#         plt.imshow(image)
#         plt.title(label)
#         plt.show()
        
if __name__ == '__main__':
    dataset = CC3MDataset(
        pickle_file='../LLaVA-CC3M-Pretrain-595K/preprocessed_image_text_pairs.pkl',
        root_dir='../LLaVA-CC3M-Pretrain-595K/images',
        transform=None
    )

    print(f"Size of dataset : {len(dataset)}")

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    print(dataset[0])
    print(dataset[0][0].shape)
    print(dataset[0][1][0].shape)
    print(dataset[0][1][0])
    print(dataset[0][1][1].shape)

    for images, (input_ids, attn_mask) in dataloader:
        print(images.shape)
        print(input_ids.shape)
        print(attn_mask.shape)
        break
    
        
    
