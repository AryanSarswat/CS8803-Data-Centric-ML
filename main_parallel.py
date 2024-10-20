from contextlib import contextmanager
from itertools import chain
import json
import multiprocessing
import os
import time
from pathlib import Path
import logging

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch import distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record

import wandb
import tqdm
from dataloader.cc3m_dataloader import CC3MDataset
from models.sigclip import SigCLIP, sigclip_loss
from models.text_encoder import TextEncoder
from models.resnet_vision_encoder import ResNet25

torch.backends.cudnn.benchmark = True
LOGGER = logging.getLogger(__name__)

def  _load_dataset():
    dataset = CC3MDataset(
        pickle_file='../LLaVA-CC3M-Pretrain-595K/preprocessed_image_text_pairs.pkl',
        root_dir='../LLaVA-CC3M-Pretrain-595K/images',
        transform=None
    )

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset

@contextmanager
def rank0_first():
    rank = dist.get_rank()
    if rank == 0:
        yield
    dist.barrier()
    if rank > 0:
        yield
    dist.barrier()
    
def main():
    # Hyperparameters
    EPOCHS = 15
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-7
    NUM_WORKERS = 20
    LOG_WANDB = True
    PROJECT_NAME = "sigclip"
    EXPERIMENT_NAME = "baseline"
    
    dist.init_process_group()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == torch.cuda.device_count()
    
    logging.basicConfig(
        format=f"[rank={rank}] [%(asctime)s] %(levelname)s:%(message)s",
        level=logging.INFO,
    )

    LOGGER.info(os.environ)
    LOGGER.info(args)
    LOGGER.info(f"rank={rank} world size={world_size}")
    
    device = torch.device(f"cuda:{rank}")
    dtype = torch.bfloat16
    # NOTE: this is necessary to call before calling any other distributed code
    torch.cuda.set_device(device)
    
    with rank0_first():
        train_dataset, val_dataset = _load_dataset()
        image_encoder = ResNet25(1000, include_fc=False)
        text_encoder = TextEncoder(model_name="distilbert-base-uncased", pretrained=True)
        model = SigCLIP(image_encoder=image_encoder, text_encoder=text_encoder).to(device=device, dtype=dtype)

    LOGGER.info(f"{sum(p.numel() for p in model.parameters())} model parameters")
    model = DistributedDataParallel(model, device_ids=[rank], output_device=rank)

    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, pin_memory=True, num_workers=NUM_WORKERS)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler, pin_memory=True, num_workers=NUM_WORKERS)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = sigclip_loss
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    resumed = False
    dist.barrier()
    
    if rank == 0:
        wandb.init(project=PROJECT_NAME, name=EXPERIMENT_NAME, config= {
            "EPOCHS": EPOCHS,
            "BATCH_SIZE": BATCH_SIZE,
            "LEARNING_RATE": LEARNING_RATE,
            "WEIGHT_DECAY": WEIGHT_DECAY,
            "NUM_WORKERS": NUM_WORKERS,
            "WORLD_SIZE": world_size,
        })
        
    state = {
        "epoch": 0,
        "global_step": 0,
        "epoch_step": 0,
        "running_loss": 0,
    }
    
    for state['epoch'] in range(0, EPOCHS):
        LOGGER.info(f"Begin Epoch {state['epoch']} at step {state['global_step']}")
        progress_bar = tqdm.tqdm(len(train_dataloader), disable=rank > 0)
        
        if state['epoch_step'] > 0:
            progress_bar.update(state['epoch_step'])
        
        batches = iter(train_dataloader)
        
        for i_step in range(len(train_dataloader)):
            batch = next(batches)
            batch = {k: v.to(device=device, dtype=dtype) for k, v in batch.items()}
            
            logits = model(batch['image'], batch['input_ids'], batch['attention_mask'])
            
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(logits)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            state['global_step'] += 1
            state['epoch_step'] += 1
            state['running_loss'] += loss.item()
            progress_bar.update(1)
            
            if state['global_step'] % 100 == 0:
                info = {
                    "global_step": state['global_step'],
                    "lr": scheduler.get_last_lr()[0],
                    "loss": state['running_loss'] / state['epoch_step'],
                    "epoch": state['epoch'],
                }
                
                LOGGER.info(info)
                
                if rank == 0:
                    wandb.log(info, step=state['global_step'])
                
                state['running_loss'] = 0
                
            if state['global_step'] % 1000 == 0:
                if rank == 0:
                    torch.save(model.state_dict(), f"saved_models/sigclip_baseline_{state['global_step']}.pth")
                dist.barrier()
                
        batches = iter(val_dataloader)
        
        with torch.no_grad():
            for i_step in range(len(val_dataloader)):
                batch = next(batches)
                batch = {k: v.to(device=device, dtype=dtype) for k, v in batch.items()}
                
                logits = model(batch['image'], batch['input_ids'], batch['attention_mask'])
                loss = criterion(logits)
                
                state['running_loss'] += loss.item()
                
            val_loss = state['running_loss'] / len(val_dataloader)
            state['running_loss'] = 0
            
            if rank == 0:
                wandb.log({"val_loss": val_loss}, step=state['global_step'])
                
            LOGGER.info(f"Validation Loss: {val_loss}")
        
        state['epoch_step'] = 0
    
    if rank == 0:
        torch.save(model.state_dict(), "saved_models/sigclip_baseline.pth")
        wandb.finish()
        
if __name__ == "__main__":
    main()
    