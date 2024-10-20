import torch
from torchivsion.models import resnet18, resnet34, resnet50, resnet101, resnet152, ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights
from torchsummary import summary
        
def ResNet18(num_classes, pretrained=True, include_fc=True):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    
    if not include_fc:
        model.fc = torch.nn.Identity()
        
    return model
    
def ResNet50(include_fc=True):
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    
    if not include_fc:
        model.fc = torch.nn.Identity()
        
    return model

def ResNet101(include_fc=True):
    model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
    
    if not include_fc:
        model.fc = torch.nn.Identity()
        
    return model

def ResNet152(include_fc=True):
    model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)
    
    if not include_fc:
        model.fc = torch.nn.Identity()
        
    return model


if __name__ == "__main__":
    model = ResNet50()
    summary(model, (3, 224, 224))
    