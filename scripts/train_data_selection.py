import argparse
import torch
import torch.optim as optim
from copy import deepcopy
import wandb

from tqdm import tqdm

from torch.utils.data import DataLoader, random_split

from dataloader.coco_dataloader import get_coco_dataloader, get_coco_train_subset_dataloader
from models.vit_vision_encoder import vit_base

from scripts.train import Trainer
from prune.prune import unstructured_prune_model, apply_global_structured_pruning
from prune.quantize import quantize
from data_selection.data_selection_techniques import least_confidence, entropy_based, least_confidence_batch, entropy_based_batch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_data_folder', default='../coco/images', type=str)
    parser.add_argument('--pickle_folder', default='./', type=str)
    parser.add_argument('--epochs', default=2, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--weight_decay', default=1e-6, type=float)
    parser.add_argument('--num_workers', default=8, type=int)
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

class TrainerDataSelection:
    def __init__(self, 
                 args,
                 data_selection_technique,
                 target_model, 
                 target_trainer,
                 prune_type,
                 prune_ratio, 
                 is_quantized,
                 s0_pool_percent, 
                 budget_percent,
                 device, 
                 project_name="", 
                 experiment_name="", 
                 test_script=False) -> None:
        self.args = args
        self.data_selection_technique = data_selection_technique
        self.target_model = target_model
        self.target_trainer = target_trainer
        self.prune_type = prune_type
        self.prune_ratio = prune_ratio
        self.is_quantized = is_quantized
        self.s0_pool_percent = s0_pool_percent
        self.budget_percent = budget_percent
        self.device = device
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.test_script = test_script
        
    def train_data_selection_epoch(self, dataloader):
        """
        Trains the model for one epoch.

        Args:
            dataloader (DataLoader): DataLoader providing training data.

        Returns:
            float: Average loss for the epoch.
        """
    
        # **Train target model on s0_percent subset of train data**
        coco_train_dataset = dataloader.dataset

        s0_num_examples = round(self.s0_pool_percent*len(coco_train_dataset))
        
        all_idxs =  torch.arange(len(coco_train_dataset))     
        s0_idxs = torch.randperm(len(coco_train_dataset))[:s0_num_examples]
        n_idxs = all_idxs[~torch.isin(all_idxs, s0_idxs)] 

        s0_dataloader = get_coco_train_subset_dataloader(self.args, s0_idxs)

        s0_loss = self.target_trainer.train_epoch(s0_dataloader)
        if self.target_trainer.wandb_log:
            metrics = {"s0_loss": s0_loss }
            wandb.log(metrics)
        print(f"s0_loss {s0_loss}")


        # **Proxy model: pruning**
        if self.prune_type == "unstructured":
            proxy_model = unstructured_prune_model(deepcopy(self.target_model), self.prune_ratio)
        elif self.prune_type == "structured":
            proxy_model = apply_global_structured_pruning(deepcopy(self.target_model), self.prune_ratio, vit=True) 

        proxy_model.to(self.device)

        # **Proxy model: quantization**
        #if self.is_quantized:
            #proxy_model = quantize(proxy_model)
            #proxy_model.to(self.device)

        # **Data selection of self.budget_percent 
        # or b data points from proxy model**
        b = round(self.budget_percent*len(coco_train_dataset))
        n_dataloader = get_coco_train_subset_dataloader(self.args, n_idxs)
        proxy_model.eval()

        scores = []

        with torch.no_grad():
            for i, data in enumerate(tqdm(n_dataloader, desc="Validating")):
                images, labels, idxs = data
                images = images.to(self.args.device, non_blocking=True)
                labels = labels.to(self.args.device, non_blocking=True)

                logits = proxy_model(images)

                prob_dist = torch.nn.functional.softmax(logits, dim=1)
                
                if self.data_selection_technique == "least_confidence":
                    batch_scores  = least_confidence_batch(prob_dist)
                    scores.extend(list(zip(idxs.cpu().numpy(), batch_scores.cpu().numpy())))
                elif self.data_selection_technique == "max_entropy":
                    batch_scores  = entropy_based_batch(prob_dist)
                    scores.extend(list(zip(idxs.cpu().numpy(), batch_scores.cpu().numpy())))                      
  
        if self.data_selection_technique == "least_confidence":
            scores.sort(key=lambda x: x[1]) # Sort least confident first
            b_idxs = [item[0] for item in scores[:b]]  # Select b
        elif self.data_selection_technique == "max_entropy":
            scores.sort(key=lambda x: x[1], reverse=True) # Sort max entropy first
            b_idxs = [item[0] for item in scores[:b]]  # Select b

        # **Train target model on b points**
        b_dataloader = get_coco_train_subset_dataloader(self.args, b_idxs)
        
        b_loss = self.target_trainer.train_epoch(b_dataloader)
        return b_loss


    def train_data_selection(self, train_dataloader, val_dataloader, epochs):
        """
        Trains the model over multiple epochs on the dataset, evaluates on validation data

        Args:
            train_dataloader (DataLoader): DataLoader providing training data.
            val_dataloader (DataLoader): DataLoader providing validation data.
            epochs (int): Number of epochs to train the model.

        Returns:
            None
        """
        for epoch in range(epochs):
            train_loss = self.train_data_selection_epoch(train_dataloader)
            val_loss, accuracy, recall, precision, f1 = self.target_trainer.evaluate(val_dataloader)

            if self.target_trainer.wandb_log:
                metrics = {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "accuracy": accuracy,
                    "recall": recall,
                    "precision": precision,
                    "f1": f1
                }

                wandb.log(metrics)
            
            print(f"Epoch: {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%, Recall: {recall:.2f}%, Precision: {precision:.2f}%, F1: {f1:.2f}%")
            
            self.target_trainer.scheduler.step(val_loss)

def test_data_selection():
    # **Configuration Arguments**
    args = parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # **Load Dataset**
    train_dataloader, val_dataloader = get_coco_dataloader(args)

    # **Initialize Model**
    target_model = vit_base(num_classes=80).to(args.device)
    # **Define Criterion, Optimizer, and Scheduler**
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        target_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # **Initialize Trainer**
    target_trainer = Trainer(
        model=target_model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=args.device,
        wandb_log=True,  # Set to True to enable Weights & Biases logging
        project_name=args.project_name,
        experiment_name=args.experiment_name,
        test_script=True,  # Set to True to limit the number of batches during testing
    )

    # **Initialize Trainer with Data Selection**
    trainer_with_data_selection = TrainerDataSelection(
        args=args,
        data_selection_technique="least_confidence",  # arg
        target_model=target_model, 
        target_trainer=target_trainer,
        prune_type="structured", # arg
        prune_ratio =0.5,  # arg
        is_quantized=False, # arg
        s0_pool_percent=0.05,  # arg
        budget_percent=0.45, # arg
        device=args.device, 
        project_name=args.project_name,
        experiment_name=args.experiment_name,
        test_script=True) 

    # **Start Training With Data Selection**
    trainer_with_data_selection.train_data_selection(train_dataloader, val_dataloader, 1)

if __name__ == "__main__":
    test_data_selection()