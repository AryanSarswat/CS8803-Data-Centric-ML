import torch
import torch.optim as optim
import wandb
from model.resnet_vision_encoder import resnet50
from model.text_encoder import TextEncoder
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from tqdm import tqdm

from dataloader.cc3m_dataloader import CC3MDataset
from models.sigclip import SigCLIP, siglip_loss


class Trainer:
    def __init__(self, model, optimizer, criterion, scheduler, wandb_log=False, project_name="", experiment_name="") -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.wandb_log = wandb_log
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model.to(self.device)
        
        if self.wandb_log:
            wandb.init(project=self.project_name, name=self.experiment_name)
            wandb.watch(self.model)
    
    def train_epoch(self, dataloader):
        self.model.train()
        running_loss = 0.0
        for i, data in enumerate(tqdm(dataloader, desc="Training")):
            ids, image, text = data
            ids, image, text = image.to(self.device, non_blocking=True), text.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(image, text)
            loss = self.criterion(logits)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        
        return running_loss / len(dataloader)
    
    def evaluate(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(tqdm(dataloader, desc="Validation")):
                ids, image, text = data
                ids, image, text = image.to(self.device, non_blocking=True), text.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                
                logits = self.model(image, text)
                loss = self.criterion(logits)
                running_loss += loss.item()
        
        return running_loss / len(dataloader)
    
    def train(self, train_dataloader, val_dataloader, epochs):
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_dataloader)
            val_loss = self.evaluate(val_dataloader)
            
            if self.wandb_log:
                wandb.log({"train_loss": train_loss, "val_loss": val_loss})
            
            print(f"Epoch: {epoch+1}/{epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}")
            self.scheduler.step(val_loss)
            
if __name__ == "__main__":
    dataset = CC3MDataset(
        csv_file='LLaVA-CC3M-Pretrain-595K/metadata.json',
        root_dir='LLaVA-CC3M-Pretrain-595K/images',
        transform=ToTensor()
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    
    image_encoder = resnet50(1000, include_fc=False)
    text_encoder = TextEncoder(model_name="distilbert-base-uncased", pretrained=True)
    model = SigCLIP(image_encoder=image_encoder, text_encoder=text_encoder)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    criterion = siglip_loss
    
    trainer = Trainer(model, optimizer, criterion, scheduler, wandb_log=False, project_name="sigclip", experiment_name="cc3m")
    trainer.train(train_dataloader, val_dataloader, epochs=3)