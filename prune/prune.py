import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import time

from models.vit_vision_encoder import vit_50M
from models.text_encoder import TextEncoder
from models.sigclip import SigCLIP
from copy import deepcopy

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

def apply_global_structured_pruning(model, pruning_amount=0.2):
    """
    Applies global structured pruning to the model by removing entire filters or neurons.

    Args:
        model (nn.Module): The PyTorch model to prune.
        pruning_amount (float): The fraction of structures to prune globally (0 < pruning_amount < 1).

    Returns:
        nn.Module: The pruned model.
    """
    # Collect all weights and their L1 norms for structured pruning
    importance_scores = []
    parameters_to_prune = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Calculate L1 norm of each filter (out_channels)
            weight = module.weight.data.abs().sum(dim=(1, 2, 3))
            importance_scores.extend(weight.tolist())
            parameters_to_prune.append((module, 'weight'))
        elif isinstance(module, nn.Linear):
            # Calculate L1 norm of each neuron (out_features)
            weight = module.weight.data.abs().sum(dim=1)
            importance_scores.extend(weight.tolist())
            parameters_to_prune.append((module, 'weight'))

    # Determine the threshold for pruning based on global importance scores
    threshold = torch.quantile(torch.tensor(importance_scores), pruning_amount)

    # Apply structured pruning based on the threshold
    for module, param_name in parameters_to_prune:
        weight = getattr(module, param_name)
        if isinstance(module, nn.Conv2d):
            mask = weight.data.abs().sum(dim=(1, 2, 3)) > threshold
            epsilon = 1e-8
            amount = max(1 - mask.float().mean(), epsilon)
            prune.ln_structured(module, name=param_name, amount=amount, n=1, dim=0)
        elif isinstance(module, nn.Linear):
            mask = weight.data.abs().sum(dim=1) > threshold
            prune.ln_structured(module, name=param_name, amount=1-mask.float().mean(), n=1, dim=0)

        prune.remove(module, param_name)

    # Calculate pruning statistics
    total_params, non_zero_params = count_parameters(model)
    pruning_ratio = 100. * (total_params - non_zero_params) / total_params
    print(f"Total Parameters: {total_params}")
    print(f"Non-zero Parameters after Pruning: {non_zero_params}")
    print(f"Pruning Ratio: {pruning_ratio:.2f}%")

    return model


if __name__ == "__main__":
    image_encoder = vit_50M(num_classes=1000, include_fc=False)
    text_encoder = TextEncoder(model_name="distilbert-base-uncased")
    model = SigCLIP(image_encoder=image_encoder, text_encoder=text_encoder)
    
    prune_ratio = 0.5
    model_unstructured = unstructured_prune_model(deepcopy(model), prune_ratio)
    model_structured = apply_global_structured_pruning(deepcopy(model), pruning_amount=prune_ratio)
    
    print("Unstructured Pruning:")
    check_sparsity(model_unstructured)
    measure_inference_speed(model_unstructured)
    
    print("\nStructured Pruning:")
    check_sparsity(model_structured)
    measure_inference_speed(model_structured)
    
    
    
    

        