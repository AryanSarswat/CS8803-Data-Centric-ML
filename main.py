import os
import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from dataloader.cc3m_dataloader import CC3MDataset
from models.resnet_vision_encoder import ResNet50
from models.sigclip import SigCLIP, sigclip_loss
from models.text_encoder import TextEncoder
from scripts.train import Trainer
from scripts.test import zero_shot_classification_pipeline

torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='../LLaVA-CC3M-Pretrain-595K')
    args = parser.parse_args()
    return args

def baseline():
    # Hyperparameters
    EPOCHS = 20
    BATCH_SIZE = 4096 #128
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 1e-6
    NUM_WORKERS = 20
    LOG_WANDB = True
    PROJECT_NAME = "sigclip"
    EXPERIMENT_NAME = "baseline_pretrained_frozen"
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = parse_args()

    # Load the dataset
    dataset = CC3MDataset(
        pickle_file=os.path.join(args.dataset, 'preprocessed_image_text_pairs.pkl'),
        root_dir=os.path.join(args.dataset, 'images'),
        transform=None
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=NUM_WORKERS)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=NUM_WORKERS)
    
    # Load the model
    image_encoder = ResNet50(include_fc=False)
    text_encoder = TextEncoder(model_name="distilbert-base-uncased", pretrained=True)
    model = SigCLIP(image_encoder=image_encoder, text_encoder=text_encoder)
    model.freeze_image_encoder()
    model.freeze_text_encoder()
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = sigclip_loss
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    trainer = Trainer(model, optimizer, criterion, scheduler, LOG_WANDB, PROJECT_NAME, EXPERIMENT_NAME, freeze_backbones=True)
    
    trainer.train(train_dataloader, val_dataloader, EPOCHS)
    
    # Save the model
    model_dir = "saved_models"
    os.makedirs(model_dir, exist_ok=True)
    model.save(model_path)

    # Do Zero Shot Classification

    class_names = ['airplanes', 'cars', 'birds', 'cats', 'deer', 'dogs', 'frogs', 'horses', 'ships', 'trucks']

    zero_shot_final = zero_shot_classification_pipeline(
        model_path=model,
        class_names=class_names,
        batch_size=128,
        num_workers=NUM_WORKERS,
        device=DEVICE
    )

    print(f"Final Zero-shot Accuracy on CIFAR-10 is {zero_shot_final}")
    
if __name__ == "__main__":
    baseline()