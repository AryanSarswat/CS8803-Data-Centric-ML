import argparse
import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor

from dataloader.cc3m_dataloader import CC3MDataset
from dataloader.cifar_dataloader import get_cifar10_classes
from dataloader.imagenet_dataloader import get_imagenet_classes
from models.resnet_vision_encoder import ResNet50
from models.sigclip import SigCLIP, sigclip_loss
from models.text_encoder import TextEncoder
from scripts.test import zero_shot_classification_pipeline
from scripts.train import Trainer

torch.backends.cudnn.benchmark = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dataset', default='cifar10', type=str, help='Dataset to test on (CIFAR-10, ImageNet)')
    parser.add_argument('--data_folder', default='..', type=str)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--weight_decay', default=1e-6, type=float)
    parser.add_argument('--num_workers', default=20, type=int)
    parser.add_argument('--log_wandb', default=False, type=bool)
    parser.add_argument('--project_name', default='sigclip', type=str)
    parser.add_argument('--experiment_name', default='', type=str)
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
    tfs = torch.nn.Sequential(
                             transforms.Resize((224,224)),
                             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))

    dataset = CC3MDataset(
        pickle_file=os.path.join(args.data_folder, 'preprocessed_image_text_pairs.pkl'),
        root_dir=os.path.join(args.data_folder, 'images'),
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
    model = SigCLIP(image_encoder=image_encoder, text_encoder=text_encoder, proj_dim=1024)
    model.freeze_image_encoder()
    model.freeze_text_encoder()
    model.to(args.device)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = sigclip_loss
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    get_classes = {
        'cifar10': get_cifar10_classes,
        'imagenet': get_imagenet_classes
    }

    args.class_names = get_classes[args.test_dataset](args)

    trainer = Trainer(model=model, 
                      optimizer=optimizer, 
                      criterion=criterion, 
                      scheduler=scheduler,
                      device=args.device,
                      wandb_log=LOG_WANDB, 
                      project_name=PROJECT_NAME, 
                      experiment_name=EXPERIMENT_NAME, 
                      test_script=False,
                      zero_shot_dataset=args.test_dataset,
                      zero_shot_class_names=args.class_names)
    
    trainer.train(train_dataloader, val_dataloader, EPOCHS)
    
    # Save the model
    model_dir = args.save_dir
    os.makedirs(model_dir, exist_ok=True)
    model.save(model_dir)

    # Do Zero Shot Classification

    zero_shot_final = zero_shot_classification_pipeline(
        model=model,
        class_names=args.class_names,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        device=args.device
    )

    print(f"Final Zero-shot Accuracy on CIFAR-10 is {zero_shot_final}")
    
if __name__ == "__main__":
    baseline()
