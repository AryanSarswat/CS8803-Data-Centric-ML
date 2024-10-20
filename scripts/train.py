import torch
import torch.optim as optim
import wandb
from models.resnet_vision_encoder import ResNet50
from models.text_encoder import TextEncoder
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from tqdm import tqdm

from dataloader.cc3m_dataloader import CC3MDataset
from models.sigclip import SigCLIP, sigclip_loss

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
            images, (input_ids, attention_mask) = data
            input_ids = input_ids.squeeze(1)
            attention_mask = attention_mask.squeeze(1)

            images = images.to(self.device, non_blocking=True)
            input_ids = input_ids.to(self.device, non_blocking=True)
            attention_mask = attention_mask.to(self.device, non_blocking=True)

            logits = self.model(images, input_ids, attention_mask)
            
            self.optimizer.zero_grad(set_to_none=True)
            loss = self.criterion(logits)
            loss.backward()
            self.optimizer.step()
            l_ = loss.item()
            running_loss += l_
            
            if self.wandb_log:
                wandb.log({"batch_loss": l_})

        return running_loss / len(dataloader)
    
    def evaluate(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(tqdm(dataloader, desc="Validation")):
                images, (input_ids, attention_mask) = data
                input_ids = input_ids.squeeze(1)
                attention_mask = attention_mask.squeeze(1)

                images = images.to(self.device, non_blocking=True)
                input_ids = input_ids.to(self.device, non_blocking=True)
                attention_mask = attention_mask.to(self.device, non_blocking=True)

                logits = self.model(images, input_ids, attention_mask)
                loss = self.criterion(logits)
                running_loss += loss.item()

        return running_loss / len(dataloader)
    
    def train(self, train_dataloader, val_dataloader, epochs):
        best_val_loss = 1e100
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_dataloader)
            val_loss = self.evaluate(val_dataloader)
            
            if self.wandb_log:
                wandb.log({"train_loss": train_loss, "val_loss": val_loss})
            
            print(f"Epoch: {epoch+1}/{epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}")
            self.scheduler.step(val_loss)

            if val_loss < best_val_loss:
                val_loss = best_val_loss
                self.model.save("./saved_model.pth")
            
if __name__ == "__main__":
    dataset = CC3MDataset(
        pickle_file='../LLaVA-CC3M-Pretrain-595K/preprocessed_image_text_pairs.pkl',
        root_dir='../LLaVA-CC3M-Pretrain-595K/images',
        transform=None
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, pin_memory=True, num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False, pin_memory=True, num_workers=8)
        
    image_encoder = ResNet50(include_fc=False).to(device)
    text_encoder = TextEncoder(model_name="distilbert-base-uncased", device=device, pretrained=True).to(device)
    model = SigCLIP(image_encoder=image_encoder, text_encoder=text_encoder)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    criterion = siglip_loss
    
    trainer = Trainer(model, optimizer, criterion, scheduler, wandb_log=False, project_name="sigclip", experiment_name="cc3m")
    trainer.train(train_dataloader, val_dataloader, epochs=2)