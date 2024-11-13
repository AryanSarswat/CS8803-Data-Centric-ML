import argparse
import os
import wandb
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
from scripts.train_data_selection import TrainerDataSelection

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
    parser.add_argument('--experiment_name', default='data_selection', type=str)
    parser.add_argument('--save_dir', default='saved_models', type=str)

    parser.add_argument('--data_selection_technique', default='least_confidence', type=str)
    parser.add_argument('--prune_type', default='unstructured', type=str)
    parser.add_argument('--prune_ratio', default=0.5, type=float)
    parser.add_argument('--is_quantized', default=False, type=bool)
    parser.add_argument('--s0_pool_percent', default=0.05, type=float)
    parser.add_argument('--budget_percent', default=0.45, type=float)
    
    args = parser.parse_args()
    return args

def data_selection_model():
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

    # Other hyperparameters
    DATA_SELECTION_TECHNIQUE = args.data_selection_technique
    PRUNE_TYPE = args.prune_type
    PRUNE_RATIO = args.prune_ratio
    IS_QUANTIZED = args.is_quantized
    S0_POOL_PERCENT = args.s0_pool_percent
    BUDGET_PERCENT = args.budget_percent

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the dataset
    train_dataloader, val_dataloader = get_coco_dataloader(args)

    # Load the model
    model = vit_50M(num_classes=80).to(args.device)

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

    # initialize data selection trainer
    trainer_with_data_selection = TrainerDataSelection(
        args=args,
        data_selection_technique=DATA_SELECTION_TECHNIQUE, 
        target_model=model, 
        target_trainer=trainer,
        prune_type=PRUNE_TYPE,
        prune_ratio =PRUNE_RATIO,
        is_quantized=IS_QUANTIZED,
        s0_pool_percent=S0_POOL_PERCENT, 
        budget_percent=BUDGET_PERCENT,
        device=args.device, 
        project_name=args.project_name,
        experiment_name=args.experiment_name,
        test_script=False) 

    # **Start Training With Data Selection**
    trainer_with_data_selection.train_data_selection(train_dataloader, val_dataloader, EPOCHS)
    
    # Save the model
    model_dir = args.save_dir
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, EXPERIMENT_NAME + '.pth'))

    
if __name__ == "__main__":
    data_selection_model()
