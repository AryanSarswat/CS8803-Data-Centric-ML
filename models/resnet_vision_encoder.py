import torch
from torchsummary import summary
from torchvision.models import (ResNet18_Weights, ResNet34_Weights,
                                ResNet50_Weights, ResNet101_Weights,
                                ResNet152_Weights, resnet18, resnet34,
                                resnet50, resnet101, resnet152)


def ResNet18(include_fc=True, num_classes=None):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    
    if not include_fc:
        model.fc = torch.nn.Identity()
    
    if num_classes is not None:
        model.fc = torch.nn.Linear(512, num_classes)

    return model
    
def ResNet50(include_fc=True, num_classes=None):
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    
    if not include_fc:
        model.fc = torch.nn.Identity()

    if num_classes is not None:
        model.fc = torch.nn.Linear(2048, num_classes)
        
    return model

def ResNet101(include_fc=True, num_classes=None):
    model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
    
    if not include_fc:
        model.fc = torch.nn.Identity()

    if num_classes is not None:
        model.fc = torch.nn.Linear(2048, num_classes)

    return model

def ResNet152(include_fc=True, num_classes=None):
    model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)
    
    if not include_fc:
        model.fc = torch.nn.Identity()

    if num_classes is not None:
        model.fc = torch.nn.Linear(2048, num_classes)

    return model


if __name__ == "__main__":
    model = ResNet152(include_fc=False, num_classes=80).cuda()
    summary(model, (3, 224, 224))
    