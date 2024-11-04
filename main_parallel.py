import json
import logging
import multiprocessing
import os
import time
from contextlib import contextmanager
from itertools import chain
from pathlib import Path

import torch
import tqdm
import wandb
from torch import distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataloader.imagenet_dataloader import get_imagenet_dataloader
from models.vit_vision_encoder import vit_50M

torch.backends.cudnn.benchmark = True
LOGGER = logging.getLogger(__name__)

@contextmanager
def rank0_first():
    rank = dist.get_rank()
    if rank == 0:
        yield
    dist.barrier()
    if rank > 0:
        yield
    dist.barrier()

def accuracy(output, target, topk=(1, 5)):
    """
    Computes the top-k accuracies for the specified values of k.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # Get top k predictions
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        # Compare with target
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res  # List of accuracies for each k

def main():
    # Hyperparameters
    EPOCHS = 15
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-7
    NUM_WORKERS = 20
    LOG_WANDB = True
    PROJECT_NAME = "ImageNet_DML"
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
    LOGGER.info(f"rank={rank} world size={world_size}")

    device = torch.device(f"cuda:{rank}")
    dtype = torch.bfloat16
    torch.cuda.set_device(device)

    with rank0_first():
        train_dataset, val_dataset = get_imagenet_dataloader()

    model = vit_50M().to(device)
    LOGGER.info(f"{sum(p.numel() for p in model.parameters())} model parameters")
    model = DistributedDataParallel(model, device_ids=[rank], output_device=rank)

    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        pin_memory=True,
        num_workers=NUM_WORKERS
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        sampler=val_sampler,
        pin_memory=True,
        num_workers=NUM_WORKERS
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True
    )

    resumed = False
    dist.barrier()

    if rank == 0 and LOG_WANDB:
        wandb.init(
            project=PROJECT_NAME,
            name=EXPERIMENT_NAME,
            config={
                "EPOCHS": EPOCHS,
                "BATCH_SIZE": BATCH_SIZE,
                "LEARNING_RATE": LEARNING_RATE,
                "WEIGHT_DECAY": WEIGHT_DECAY,
                "NUM_WORKERS": NUM_WORKERS,
                "WORLD_SIZE": world_size,
            }
        )

    state = {
        "epoch": 0,
        "global_step": 0,
        "epoch_step": 0,
        "running_loss": 0,
        "running_top1": 0,
        "running_top5": 0,
        "val_running_loss": 0,
        "val_running_top1": 0,
        "val_running_top5": 0,
    }

    for state['epoch'] in range(EPOCHS):
        LOGGER.info(f"Begin Epoch {state['epoch']} at step {state['global_step']}")
        progress_bar = tqdm.tqdm(len(train_dataloader), disable=rank > 0)

        if state['epoch_step'] > 0:
            progress_bar.update(state['epoch_step'])

        batches = iter(train_dataloader)

        for _ in range(len(train_dataloader)):
            batch = next(batches)
            images, labels = batch
            images = images.to(device=device, dtype=dtype)
            labels = labels.to(device=device)

            logits = model(images)

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

            # Compute top-1 and top-5 accuracies
            top1, top5 = accuracy(logits, labels, topk=(1, 5))
            state['running_top1'] += top1.item()
            state['running_top5'] += top5.item()

            state['global_step'] += 1
            state['epoch_step'] += 1
            state['running_loss'] += loss.item()
            progress_bar.update(1)

            if state['global_step'] % 100 == 0:
                avg_loss = state['running_loss'] / state['epoch_step']
                avg_top1 = state['running_top1'] / state['epoch_step']
                avg_top5 = state['running_top5'] / state['epoch_step']

                info = {
                    "global_step": state['global_step'],
                    "lr": scheduler.optimizer.param_groups[0]['lr'],
                    "loss": avg_loss,
                    "top1_acc": avg_top1,
                    "top5_acc": avg_top5,
                    "epoch": state['epoch'],
                }

                LOGGER.info(info)

                if rank == 0 and LOG_WANDB:
                    wandb.log(info, step=state['global_step'])

                state['running_loss'] = 0
                state['running_top1'] = 0
                state['running_top5'] = 0

            if state['global_step'] % 1000 == 0:
                if rank == 0:
                    Path("saved_models").mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), f"saved_models/vit50m_baseline_{state['global_step']}.pth")
                dist.barrier()

        # Validation
        batches = iter(val_dataloader)
        state['val_running_loss'] = 0
        state['val_running_top1'] = 0
        state['val_running_top5'] = 0

        with torch.no_grad():
            for _ in range(len(val_dataloader)):
                batch = next(batches)
                images, labels = batch
                images = images.to(device=device, dtype=dtype)
                labels = labels.to(device=device)

                logits = model(images)
                loss = criterion(logits, labels)

                state['val_running_loss'] += loss.item()

                # Compute top-1 and top-5 accuracies
                top1, top5 = accuracy(logits, labels, topk=(1, 5))
                state['val_running_top1'] += top1.item()
                state['val_running_top5'] += top5.item()

        val_loss = state['val_running_loss'] / len(val_dataloader)
        val_top1 = state['val_running_top1'] / len(val_dataloader)
        val_top5 = state['val_running_top5'] / len(val_dataloader)

        if rank == 0 and LOG_WANDB:
            wandb.log({
                "val_loss": val_loss,
                "val_top1_acc": val_top1,
                "val_top5_acc": val_top5
            }, step=state['global_step'])

        LOGGER.info(f"Validation Loss: {val_loss:.4f}, Top-1 Acc: {val_top1:.2f}%, Top-5 Acc: {val_top5:.2f}%")

        # Reset epoch step counters
        state['epoch_step'] = 0

    if rank == 0:
        Path("saved_models").mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), "saved_models/vit50m_baseline.pth")
        if LOG_WANDB:
            wandb.finish()

if __name__ == "__main__":
    main()