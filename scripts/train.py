import torch
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision import datasets, transforms
from tqdm import tqdm
import argparse
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from dataloader.coco_dataloader import get_coco_dataloader
from models.vit_vision_encoder import vit_50M

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_data_folder', default='../coco/images', type=str)
    parser.add_argument('--pickle_folder', default='./', type=str)
    parser.add_argument('--epochs', default=2, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--weight_decay', default=1e-6, type=float)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--log_wandb', default=False, type=bool)
    parser.add_argument('--project_name', default='svp', type=str)
    parser.add_argument('--experiment_name', default='baseline_test', type=str)
    parser.add_argument('--save_dir', default='saved_models', type=str)
    
    args = parser.parse_args()
    return args

class Trainer:
    def __init__(self, 
                 model, 
                 optimizer, 
                 criterion, 
                 scheduler, 
                 device, 
                 wandb_log=False, 
                 project_name="", 
                 experiment_name="", 
                 test_script=False) -> None:
        
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.wandb_log = wandb_log
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.device = device
        self.test_script = test_script
        
        if self.wandb_log:
            wandb.init(project=self.project_name, name=self.experiment_name)

    def train_epoch(self, dataloader):
        """
        Trains the model for one epoch.

        Args:
            dataloader (DataLoader): DataLoader providing training data.

        Returns:
            float: Average loss for the epoch.
        """
        self.model.train()
        
        running_loss = 0.0
        for i, data in enumerate(tqdm(dataloader, desc="Training")):
            images, labels, _ = data
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            logits = self.model(images)
            
            self.optimizer.zero_grad(set_to_none=True)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()
            l_ = loss.item()
            running_loss += l_

            if self.test_script and i == 10:
                break

            if self.wandb_log:
                wandb.log({"batch_loss": l_})

        return running_loss / len(dataloader)
    
    def evaluate(self, dataloader, thres=0.5):
        """
        Evaluates the model on the validation dataset.

        Args:
            dataloader (DataLoader): DataLoader providing validation data.

        Returns:
            float: Average loss over the validation dataset.
            float: Accuracy.
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for i, data in enumerate(tqdm(dataloader, desc="Validating")):
                images, labels, _ = data
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                logits = self.model(images)
                loss = self.criterion(logits, labels)
                running_loss += loss.item()

                preds = (logits > thres).int().flatten()
                labels = labels.flatten()
                all_preds.append(preds)
                all_labels.append(labels)

                if self.test_script and i == 10:
                    break
        all_preds = torch.cat(all_preds).cpu()
        all_labels = torch.cat(all_labels).cpu()
        avg_loss = running_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)

        return avg_loss, accuracy, recall, precision, f1
    
    def train(self, train_dataloader, val_dataloader, epochs):
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
            train_loss = self.train_epoch(train_dataloader)
            val_loss, accuracy, recall, precision, f1 = self.evaluate(val_dataloader)

            if self.wandb_log:
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
            
            self.scheduler.step(val_loss)

def test_trainer():
    # **Configuration Arguments**
    args = parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # **Load Dataset**
    train_dataloader, val_dataloader = get_coco_dataloader(args)

    # **Initialize Model**
    model = vit_50M(num_classes=80).to(args.device)

    # **Define Criterion, Optimizer, and Scheduler**
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # **Initialize Trainer**
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=args.device,
        wandb_log=True,  # Set to True to enable Weights & Biases logging
        project_name=args.project_name,
        experiment_name=args.experiment_name,
        test_script=True,  # Set to True to limit the number of batches during testing
    )

    # **Start Training**
    trainer.train(train_dataloader, val_dataloader, epochs=args.epochs)

if __name__ == "__main__":
    test_trainer()
