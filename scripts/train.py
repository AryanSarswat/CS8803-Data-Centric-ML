import torch
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from tqdm import tqdm

from dataloader.cc3m_dataloader import CC3MDataset
from models.resnet_vision_encoder import ResNet50
from models.sigclip import SigCLIP, sigclip_loss
from models.text_encoder import TextEncoder

from .test import zero_shot_classification_pipeline

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
                 test_script=False, 
                 freeze_backbones=False, 
                 zero_shot_dataset='cifar10', 
                 zero_shot_class_names=None) -> None:

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.wandb_log = wandb_log
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.device = device
        self.test_script = test_script
        self.freeze_backbones = freeze_backbones
        self.zero_shot_dataset = zero_shot_dataset
        self.zero_shot_class_names = zero_shot_class_names
        
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
        
        if self.freeze_backbones:
            self.model.freeze_image_encoder()
            self.model.freeze_text_encoder()
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

            if self.test_script and i == 10:
                break

            if self.wandb_log:
                wandb.log({"batch_loss": l_})

        return running_loss / len(dataloader)
    
    def evaluate(self, dataloader):
        """
        Evaluates the model on a validation dataset.

        Args:
            dataloader (DataLoader): DataLoader providing validation data.

        Returns:
            float: Average loss over the validation dataset.
        """
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

                if self.test_script and i == 10:
                    break

        return running_loss / len(dataloader)
    
    def train(self, train_dataloader, val_dataloader, epochs):
        """
        Trains the model over multiple epochs, evaluates on validation data, 
        and optionally performs zero-shot classification.

        Args:
            train_dataloader (DataLoader): DataLoader providing training data.
            val_dataloader (DataLoader): DataLoader providing validation data.
            epochs (int): Number of epochs to train the model.

        Returns:
            None
        """
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_dataloader)
            val_loss = self.evaluate(val_dataloader)

            if self.zero_shot_class_names is not None:
                zero_shot_acc = zero_shot_classification_pipeline(self.model, self.zero_shot_class_names)
            
            if self.wandb_log:
                wandb.log({"train_loss": train_loss, "val_loss": val_loss, f'{self.zero_shot_dataset}_zero_shot': zero_shot_acc})
            
            print(f"Epoch: {epoch+1}/{epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}, {self.zero_shot_dataset}_Zero_Shot_Accuracy : {zero_shot_acc:.2f}%")
            self.scheduler.step(val_loss)

            
            
if __name__ == "__main__":
    dataset = CC3MDataset(
        pickle_file='../LLaVA-CC3M-Pretrain-595K/preprocessed_image_text_pairs.pkl',
        root_dir='../LLaVA-CC3M-Pretrain-595K/images',
        transform=None
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = {
        'device' : device,
        'class_names' : ['airplanes', 'cars', 'birds', 'cats', 'deer', 'dogs', 'frogs', 'horses', 'ships', 'trucks']
    }

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, pin_memory=True, num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False, pin_memory=True, num_workers=8)
        
    image_encoder = ResNet50(include_fc=False)
    text_encoder = TextEncoder(model_name="distilbert-base-uncased", device=device, pretrained=True)
    model = SigCLIP(image_encoder=image_encoder, text_encoder=text_encoder).to(device)
    model.freeze_image_encoder()
    model.freeze_text_encoder()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters : {params}")
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    criterion = sigclip_loss
    
    trainer = Trainer(model=model, 
                      optimizer=optimizer, 
                      criterion=criterion, 
                      scheduler=scheduler,
                      device=args['device'],
                      wandb_log=False, 
                      project_name="sigclip", 
                      experiment_name="cc3m", 
                      test_script=True,
                      zero_shot_dataset='cifar10',
                      zero_shot_class_names=args['class_names'])

    trainer.train(train_dataloader, val_dataloader, epochs=2)