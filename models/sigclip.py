import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .text_encoder import TextEncoder
from .vit_vision_encoder import vit_50M


def get_feature_size(encoder):
    """Get the feature size from the encoder using a dummy input."""
    encoder.eval()
    dummy_input = torch.randn(1, 3, 224, 224).cuda()
    output = encoder(dummy_input)
    flat_dim = output.view(output.size(0), -1)
    return flat_dim.shape[1]

def contrastive_loss(logits):
    targets = torch.arange(logits.size(0)).to(logits.device)
    loss_images = F.cross_entropy(logits, targets)
    loss_texts = F.cross_entropy(logits.t(), targets)
    return (loss_images + loss_texts) / 2

def siglip_loss(logits):
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

        self.image_projection = torch.nn.Linear(image_mlp_dim, proj_dim)
        
        self.text_projection = torch.nn.Linear(text_mlp_dim, proj_dim)
        
        self.t_prime = nn.Parameter(torch.ones([]) * init_tau)
        self.b = nn.Parameter(torch.ones([]) * init_b)

    def forward(self, image, input_ids, attention_mask):
        image_features = self.extract_image_features(image)
        text_features = self.extract_text_features(input_ids, attention_mask)
        image_features = F.normalize(image_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)
        return image_features @ text_features.t() * self.t_prime.exp() + self.b

    def extract_image_features(self, images):
        image_features = self.image_encoder(images)
        image_features = image_features.view(image_features.size(0), -1)
        return self.image_projection(image_features)

    def extract_text_features(self, input_ids, attention_mask):
        text_features = self.text_encoder(input_ids, attention_mask)
        return self.text_projection(text_features)
    
    def save(self, path):
        torch.save(self.state_dict(), path)
        
    def load(self, path):
        self.load_state_dict(torch.load(path))

    
if __name__ == "__main__":
    image_encoder = vit_50M(num_classes=1000, include_fc=False)
    text_encoder = TextEncoder(model_name="distilbert-base-uncased")

    model = SigCLIP(image_encoder=image_encoder, text_encoder=text_encoder)
    
    print(model.extract_image_features(torch.randn(1, 3, 224, 224)).shape)
    print(model.extract_text_features_from_ids(torch.randint(0, 1, (1, 512)), torch.randn(1, 512)).shape)
    print(model.extract_text_features_from_text(["Hello, my dog is cute."]).shape)

