import argparse
import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor

from dataloader.coco_dataloader import get_coco_dataloader
from dataloader.cifar_dataloader import get_cifar10_classes
from models.resnet_vision_encoder import ResNet152
from models.vit_vision_encoder import vit_50M, vit_base
from scripts.train import Trainer

torch.backends.cudnn.benchmark = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_data_folder', default='../coco/images', type=str)
    parser.add_argument('--pickle_folder', default='./', type=str)
    parser.add_argument('--epochs', default=2, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--weight_decay', default=1e-6, type=float)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--log_wandb', default=True, type=bool)
    parser.add_argument('--project_name', default='svp', type=str)
    parser.add_argument('--experiment_name', default='baseline', type=str)
    parser.add_argument('--save_dir', default='saved_models', type=str)
    
    args = parser.parse_args()
    return args

def baseline():
    args = parse_args()
    # Hyperparameters
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    WEIGHT_DECAY = args.weight_decay
    NUM_WORKERS = args.num_workers
    LOG_WANDB = args.log_wandb
    PROJECT_NAME = args.project_name
    EXPERIMENT_NAME = args.experiment_name
    
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the dataset
    train_dataloader, val_dataloader = get_coco_dataloader(args)
    
    # Load the model
    model = ResNet152(num_classes=80).to(args.device)

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Number of trainable paramters : {num_parameters}")
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.BCEWithLogitsLoss() #torch.nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    trainer = Trainer(
                    model=model,
                    optimizer=optimizer,
                    criterion=criterion,
                    scheduler=scheduler,
                    device=args.device,
                    wandb_log=LOG_WANDB,  # Set to True to enable Weights & Biases logging
                    project_name=PROJECT_NAME,
                    experiment_name=EXPERIMENT_NAME,
                    test_script=False,  # Set to True to limit the number of batches during testing
                )
    
    trainer.train(train_dataloader, val_dataloader, EPOCHS)
    
    # Save the model
    model_dir = args.save_dir
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, EXPERIMENT_NAME + '.pth'))
    
if __name__ == "__main__":
    baseline()
