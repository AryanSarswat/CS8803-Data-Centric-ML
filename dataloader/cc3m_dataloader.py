import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import ToTensor

class CC3MDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_json(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 1])
        image = read_image(img_path)
        image = image.permute(1, 2, 0)
        label = [self.annotations.iloc[index, 2]]

        if self.transform:
            image = self.transform(image)

        return (image, label)

    def show_image(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 1])
        image = read_image(img_path)
        image = image.permute(1, 2, 0)
        label = [self.annotations.iloc[index, 2]]
        
        plt.imshow(image)
        plt.title(label)
        plt.show()
        
if __name__ == '__main__':
    dataset = CC3MDataset(
        csv_file='../LLaVA-CC3M-Pretrain-595K/metadata.json',
        root_dir='../LLaVA-CC3M-Pretrain-595K/images',
        transform=None
    )

    print(f"Size of dataset : {len(dataset)}")

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    print(dataset[0])
    print(dataset[0][0].shape)
    print(dataset[0][1])

    for images, labels in dataloader:
        print(images.shape)
        print(len(labels[0]))
        break
    
        
    
