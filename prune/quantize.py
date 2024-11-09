import time
from copy import deepcopy

import torch
import torch.nn as nn

from models.vit_vision_encoder import vit_base

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

def quantize(model):
    quantized_model = torch.ao.quantization.quantize_dynamic(model,  # the original model
                                                            {torch.nn.Linear},  # a set of layers to dynamically quantize
                                                            dtype=torch.qint8)
    
    return quantized_model

if __name__ == "__main__":
    model = vit_base(num_classes=80)
    q_model = quantize(model)

    measure_inference_speed(model)
    measure_inference_speed(q_model)