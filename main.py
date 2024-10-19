import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from dataloader.cc3m_dataloader import CC3MDataset
from models.resnet_vision_encoder import ResNet50
from models.sigclip import SigCLIP, sigclip_loss
from models.text_encoder import TextEncoder
from scripts.train import Trainer


def baseline():
    # Hyperparameters
    EPOCHS = 25
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2 
    LOG_WANDB = True
    PROJECT_NAME = "sigclip"
    EXPERIMENT_NAME = "baseline"
    
    # Load the dataset
    dataset = CC3MDataset(
        csv_file='../LLaVA-CC3M-Pretrain-595K/metadata.json',
        root_dir='../LLaVA-CC3M-Pretrain-595K/images',
        transform=ToTensor()
    )
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Load the model
    image_encoder = ResNet50(1000, include_fc=False)
    text_encoder = TextEncoder(model_name="distilbert-base-uncased", pretrained=True)
    model = SigCLIP(image_encoder=image_encoder, text_encoder=text_encoder)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = sigclip_loss
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    trainer = Trainer(model, optimizer, criterion, scheduler, LOG_WANDB, PROJECT_NAME, EXPERIMENT_NAME)
    
    trainer.train(train_dataloader, val_dataloader, EPOCHS)
    
    # Save the model
    os.makedirs("saved_models", exist_ok=True)
    model.save("saved_models/sigclip_baseline.pth")
    
if __name__ == "__main__":
    baseline()