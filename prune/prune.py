import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vit_vision_encoder import vit_1M
from models.text_encoder import TextEncoder

def unstructured_prune_model(model, prune_ratio):
    """
    _summary_ : Apply prune mask

    Args:
        model (torch.nn.Module): Model to prune
        prune_ratio (int): Prune ratio

    Returns:
        torch.nn.Module: Pruned model
    """
    total = 0
    total_nonzero = 0
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            if "mlp_head" not in name:
                total += m.weight.data.numel()
                mask = m.weight.data.abs().clone().gt(0).float().cuda()
                total_nonzero += torch.sum(mask)

    lin_weights = torch.zeros(total)
    index = 0
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            # Do not prune last MLP
            if "mlp_head" not in name:
                size = m.weight.data.numel()
                lin_weights[index:(index+size)] = m.weight.data.view(-1).abs().clone()
                index += size

    y, i = torch.sort(lin_weights)
    thre_index = total - total_nonzero + int(total_nonzero * prune_ratio)
    thre = y[int(thre_index)]
    pruned = 0
    print('Pruning threshold: {}'.format(thre))
    zero_flag = False
    for k, (name, m) in enumerate(model.named_modules()):
        if isinstance(m, nn.Linear):
            if "mlp_head" not in name:
                weight_copy = m.weight.data.abs().clone()
                mask = weight_copy.gt(thre).float().cuda()
                pruned = pruned + mask.numel() - torch.sum(mask)
                m.weight.data.mul_(mask)
                if int(torch.sum(mask)) == 0:
                    zero_flag = True
                print('layer index: {:d} \t total params: {:d} \t remaining params: {:d}'.format(k, mask.numel(), int(torch.sum(mask))))
    print('Total params: {}, Pruned params: {}, Pruned ratio: {}'.format(total, pruned, pruned/total))
    
    return model

def structured_prune_model(model, prune_ratio):
    """
    _summary_ : Prune a model layer-wise

    Args:
        model (torch.nn.Module): Model to prune
        prune_ratio (float): Prune ratio

    Returns:
        torch.nn.Module: Pruned model
    """
    pass

if __name__ == "__main__":
    image_encoder = vit_1M(num_classes=1000, include_fc=False)
    text_encoder = TextEncoder(model_name="distilbert-base-uncased", include_fc=False)
    
    model = SigCLIP(image_encoder=image_encoder, text_encoder=text_encoder)
    
    prune_ratio = 0.5
    model = unstructured_prune_model(model, prune_ratio)
    
    
    
    
    

        