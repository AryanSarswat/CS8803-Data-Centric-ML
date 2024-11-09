import time
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch_pruning as tp

from models.vit_vision_encoder import vit_base


def count_parameters(model):
    """
    Counts the total and non-zero parameters in the model.

    Args:
        model (nn.Module): The PyTorch model.

    Returns:
        total_params (int): Total number of parameters.
        non_zero_params (int): Number of non-zero parameters.
    """
    total_params = 0
    non_zero_params = 0
    for param in model.parameters():
        total_params += param.numel()
        non_zero_params += param.nonzero().size(0)

    print(f"Total number of parameters : {total_params}")
    print(f"Total number of non-zero parameters : {non_zero_params}")
    return total_params, non_zero_params

def check_sparsity(model):
    """
    Prints the sparsity of each pruned layer in the model.

    Args:
        model (nn.Module): The pruned PyTorch model.
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if hasattr(module, 'weight_mask'):
                mask = module.weight_mask
                sparsity = 100. * float(mask.nelement() - mask.sum()) / mask.nelement()
                print(f"Sparsity in {name}: {sparsity:.2f}%")

def measure_inference_speed(model, input_size=(1, 3, 224, 224), device='cpu', runs=100):
    """
    Measures the average inference time of the model.

    Args:
        model (nn.Module): The PyTorch model.
        input_size (tuple): The size of the input tensor.
        device (str): The device to run the model on ('cpu' or 'cuda').
        runs (int): Number of inference runs to average.

    Returns:
        float: Average inference time in milliseconds.
    """
    model.to(device)
    model.eval()
    dummy_input = torch.randn(input_size).to(device)
    
    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(runs):
            _ = model(dummy_input)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / runs * 1000  # milliseconds
    print(f"Average Inference Time over {runs} runs: {avg_time:.4f} ms")
    return avg_time

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
                mask = weight_copy.gt(thre).float()
                pruned = pruned + mask.numel() - torch.sum(mask)
                m.weight.data.mul_(mask)
                if int(torch.sum(mask)) == 0:
                    zero_flag = True
                print('layer index: {:d} \t total params: {:d} \t remaining params: {:d}'.format(k, mask.numel(), int(torch.sum(mask))))
    print('Total params: {}, Pruned params: {}, Pruned ratio: {}'.format(total, pruned, pruned/total))
    
    return model

def apply_global_structured_pruning(model, pruning_amount=0.2, vit=True):
    """
    Applies global structured pruning to the model by removing entire filters or neurons.

    Args:
        model (nn.Module): The PyTorch model to prune.
        pruning_amount (float): The fraction of structures to prune globally (0 < pruning_amount < 1).

    Returns:
        nn.Module: The pruned model.
    """
    example_inputs = torch.randn(1, 3, 224, 224)

    DG = tp.DependencyGraph()
    DG.build_dependency(model, example_inputs=example_inputs)
    
    imp = tp.importance.GroupNormImportance(p=2)
    ignored_layers = []

    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == 80:
            ignored_layers.append(m) # DO NOT prune the final classifier!
        if vit and isinstance(m, torch.nn.Conv2d):
            ignored_layers.append(m) # Only prune attention layers

    print(ignored_layers)
    pruner = tp.pruner.MetaPruner( # We can always choose MetaPruner if sparse training is not required.
        model,
        example_inputs,
        importance=imp,
        pruning_ratio=pruning_amount,
        ignored_layers=ignored_layers,
    )
    pruner.step()

    return model

if __name__ == "__main__":
    model = vit_base(num_classes=80)
    
    count_parameters(model)
    measure_inference_speed(model)
    
    prune_ratio = 0.5
    model_unstructured = unstructured_prune_model(deepcopy(model), prune_ratio)
    model_structured = apply_global_structured_pruning(deepcopy(model), pruning_amount=prune_ratio, vit=True)
    
    print(f"After Unstructured")
    count_parameters(model_unstructured)
    measure_inference_speed(model_unstructured)

    print(f"After Structured")
    count_parameters(model_structured)
    measure_inference_speed(model_structured)




    
    
    
    

        