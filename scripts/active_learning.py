import torch
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from dataloader.cifar_dataloader import get_cifar10_dataloader
from models.text_encoder import TextEncoder
from models.resnet_vision_encoder import ResNet25
from models.sigclip import SigCLIP, sigclip_loss
from train import Trainer

from active_learning.uncertainity_sampling import least_confidence
from active_learning.utils import set_random_seed, split_indices, validate_splits

import random
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Function to train the proxy model
def train_proxy_model(train_dataset, model):
    # Split dataset into train/validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True, pin_memory=True, num_workers=8)
    val_dataloader = DataLoader(val_data, batch_size=128, shuffle=False, pin_memory=True, num_workers=8)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    criterion = sigclip_loss

    trainer = Trainer(model, optimizer, criterion, scheduler, wandb_log=False, project_name="ActiveLearning", experiment_name="s0_train")
    trainer.train(train_dataloader, val_dataloader, epochs=2)

# Set seeds for reproducibility.

# Calculate the number of classes (e.g., 10 or 100) so the model has
#   the right dimension for its output.
num_classes = 10 # because cifar10  

# Splt Cifar10 into unlabeled_pool and dev_indices
train_loader, test_loader = get_cifar10_dataloader() 
unlabeled_pool = unlabeled_pool_loader.dataset
dev_indices = dev_indices_loader.dataset

validation = len(dev_indices)
initial_subset = 1_000
rounds = (4_000, 5_000, 5_000, 5_000, 5_000)

full_train_dataset_size = len(unlabeled_pool) + len(dev_indices)

validate_splits(full_train_dataset_size, validation, initial_subset, rounds)



## TODO: replace w/ Proxy model
image_encoder = ResNet25(1000, include_fc=False).to(device)
text_encoder = TextEncoder(model_name="distilbert-base-uncased", device=device, pretrained=True).to(device)
A0_model = SigCLIP(image_encoder=image_encoder, text_encoder=text_encoder).to(device)


# Select which points to label using the proxy.
# Create initial random subset to train the proxy (warm start).
labeled = np.random.permutation(unlabeled_pool)[:initial_subset]
#save_index(labeled, run_dir,'initial_subset_{}.index'.format(len(labeled)))

# Train the proxy on the initial random subset
model, stats = proxy_generator.send(labeled)
#utils.save_result(stats, os.path.join(run_dir, "proxy.csv"))