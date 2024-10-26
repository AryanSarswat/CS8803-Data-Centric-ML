import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .text_encoder import TextEncoder
from .vit_vision_encoder import vit_50M


def get_feature_size(encoder):
    """Get the feature size from the encoder using a dummy input."""
    encoder.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    output = encoder(dummy_input)
    flat_dim = output.view(output.size(0), -1)
    return flat_dim.shape[1]

def contrastive_loss(logits):
    targets = torch.arange(logits.size(0)).to(logits.device)
    loss_images = F.cross_entropy(logits, targets)
    loss_texts = F.cross_entropy(logits.t(), targets)
    return (loss_images + loss_texts) / 2

def sigclip_loss(logits):
    n = logits.size(0)
    # -1 for off-diagonals and 1 for diagonals
    labels = 2 * torch.eye(n, device=logits.device) - 1
    # pairwise sigmoid loss
    return -torch.sum(F.logsigmoid(labels * logits)) / n

class SigCLIP(torch.nn.Module):

    def __init__(self,
                 image_encoder,
                 text_encoder,
                 image_mlp_dim=None,
                 text_mlp_dim=768,
                 proj_dim=256,
                 init_tau=np.log(1.0),
                 init_b=0):
        super(SigCLIP, self).__init__()

        if not image_mlp_dim:
            image_mlp_dim = get_feature_size(image_encoder)

        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

        self.image_projection = nn.Sequential(
            nn.Linear(image_mlp_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
            nn.LayerNorm(proj_dim)
        )
        
        self.text_projection = nn.Sequential(
            nn.Linear(text_mlp_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
            nn.LayerNorm(proj_dim)
        )
        
        self.t_prime = nn.Parameter(torch.ones([]) * init_tau)
        self.b = nn.Parameter(torch.ones([]) * init_b)

    def forward(self, image, input_ids, attention_mask):
        image_features = self.extract_image_features(image)
        text_features = self.extract_text_features(input_ids, attention_mask)
        return image_features @ text_features.t()

    def freeze_image_encoder(self):
        for param in self.image_encoder.parameters():
            param.requires_grad = False

    def freeze_text_encoder(self):
        for param in self.text_encoder.parameters():
            param.requires_grad = False

    def extract_image_features(self, images):
        image_features = self.image_encoder(images)
        image_features = image_features.view(image_features.size(0), -1)
        return self.image_projection(image_features)

    def extract_text_features(self, input_ids, attention_mask):
        text_features = self.text_encoder(input_ids, attention_mask)
        return self.text_projection(text_features)
    
    def save(self, path):
        torch.save(self.image_encoder, os.path.join(path, "baseline_image_encoder.pth"))
        torch.save(self.text_encoder, os.path.join(path, "baseline_text_encoder.pth"))
        
    def load(self, image_encoder_path, text_encoder_path):
        self.image_encoder = torch.load(image_encoder_path, weights_only=False)
        self.text_encoder = torch.load(text_encoder_path, weights_only=False)

    
if __name__ == "__main__":
    image_encoder = vit_50M(num_classes=1000, include_fc=False)
    text_encoder = TextEncoder(model_name="distilbert-base-uncased")

    model = SigCLIP(image_encoder=image_encoder, text_encoder=text_encoder, proj_dim=5)
    

    images = torch.randn(3,3,224,224)
    texts = ["this is a dog", "this is a cat", "this is a mouse"]
    input_ids, attn_msk = model.text_encoder.tokenize(texts)
    
    image_features = model.extract_image_features(images)
    text_features = model.extract_text_features(input_ids, attn_msk)

    print(image_features)
    print(text_features)

    logits = model(images, input_ids, attn_msk)
    print(logits)

    print(sigclip_loss(logits))

