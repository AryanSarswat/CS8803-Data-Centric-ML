import os
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from lion_pytorch import Lion

from dataloader.cc3m_dataloader import CC3MDataset
from models.resnet_vision_encoder import ResNet50
from models.sigclip import SigCLIP, sigclip_loss, contrastive_loss
from models.text_encoder import TextEncoder
from scripts.train import Trainer
from scripts.test import zero_shot_classification_pipeline
from transformers import SiglipConfig, SiglipModel

torch.backends.cudnn.benchmark = True

def baseline():
    # Hyperparameters
    EPOCHS = 20
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-4 #1e-5
    WEIGHT_DECAY = 1e-7 #1e-6
    NUM_WORKERS = 20
    LOG_WANDB = True
    PROJECT_NAME = "sigclip"
    EXPERIMENT_NAME = "baseline_pretrained_frozen"
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the dataset
    # ~/scratch/DML/datasets/LLaVA-CC3M-Pretrain-595K
    dataset = CC3MDataset(
        root_dir='../datasets/LLaVA-CC3M-Pretrain-595K/',
        transform=None
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=NUM_WORKERS)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=NUM_WORKERS)
    
    # Load the model
    configuration = SiglipConfig()
    model = SiglipModel(configuration)
    model.to(DEVICE)

    # image_encoder = ResNet50(include_fc=False)
    # text_encoder = TextEncoder(model_name="distilbert-base-uncased", pretrained=True)
    # model = SigCLIP(image_encoder=image_encoder, text_encoder=text_encoder)
    # model.freeze_image_encoder()
    # model.freeze_text_encoder()
    
    # optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    optimizer = Lion(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = sigclip_loss
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # class_names = ['airplanes', 'cars', 'birds', 'cats', 'deer', 'dogs', 'frogs', 'horses', 'ships', 'trucks']
    class_names = [x[1] for x in json.load(open("data/imagenet_class_index.json")).values()]

    trainer = Trainer(model, optimizer, scheduler, class_names, LOG_WANDB, PROJECT_NAME, EXPERIMENT_NAME, freeze_backbones=False, device=DEVICE)
    
    trainer.train(train_dataloader, val_dataloader, EPOCHS)
    
    # Save the model
    model_dir = "saved_models"
    os.makedirs(model_dir, exist_ok=True)
    model.save(model_path)

    # Do Zero Shot Classification

    zero_shot_final = zero_shot_classification_pipeline(
        model,
        class_names,
    )

    print(f"Final Zero-shot Accuracy on ImageNet is {zero_shot_final}")
    
if __name__ == "__main__":
    baseline()