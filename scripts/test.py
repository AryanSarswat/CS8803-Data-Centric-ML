import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloader.cifar_dataloader import (get_cifar10_dataloader,
                                         get_cifar100_dataloader)
from models.resnet_vision_encoder import ResNet50
from models.sigclip import SigCLIP, sigclip_loss
from models.text_encoder import TextEncoder
import clip


def load_sigclip_model(model_path, device='cuda'):
    """
    Loads the trained SigCLIP model with specified weights.

    Args:
        model_path (str): Path to the saved SigCLIP model weights.
        device (str): Device on which to load the model ('cuda' for GPU or 'cpu').

    Returns:
        SigCLIP: Loaded SigCLIP model set to evaluation mode.
    """
    image_encoder = ResNet50(include_fc=False)
    text_encoder = TextEncoder(model_name="distilbert-base-uncased", pretrained=True)

    # Initialize SigCLIP
    model = SigCLIP(image_encoder, text_encoder)
    model.load(model_path)
    model.to(device)
    model.eval()
    
    return sigclip_model

def prepare_dataloader(data_folder, batch_size=32, num_workers=4, split='test'):
    """
    Prepares a DataLoader for the dataset with specified split.

    Args:
        data_folder (str): Path to the dataset folder.
        batch_size (int): Number of samples per batch in DataLoader.
        num_workers (int): Number of subprocesses to use for data loading.
        split (str): Specifies dataset split to load ('train', 'val', 'test').

    Returns:
        DataLoader: PyTorch DataLoader object for the specified dataset split.
    """
    train_dataloader, test_dataloader = get_cifar10_dataloader(data_folder=data_folder, 
                                                               batch_size=batch_size, 
                                                               num_workers=num_workers)
    
    return test_dataloader

def create_class_prompts(class_names):
    """
    Generates text prompts for each class name, formatted for CLIP-style encoding.

    Args:
        class_names (list of str): List of class names in the dataset.

    Returns:
        list of str: List of textual prompts for each class.
    """
    return [f"A photo of a {class_name}." for class_name in class_names]

def encode_prompts(prompts, model, device='cuda'):
    """
    Encodes class prompts into text embeddings using the SigCLIP model.

    Args:
        prompts (list of str): List of text prompts to be encoded.
        model (nn.Module): Loaded CLIP style model for encoding.
        device (str): Device for running computations ('cuda' for GPU or 'cpu').

    Returns:
        torch.Tensor: Normalized text embeddings for each prompt.
    """
    encoded = model.text_encoder.tokenizer(prompts, padding=True, return_tensors='pt').to(device)
    with torch.no_grad():
        text_features = model.extract_text_features(
            encoded['input_ids'],
            encoded['attention_mask']
        )
        text_features = F.normalize(text_features, p=2, dim=-1)
    return text_features

def evaluate_zero_shot(model, dataloader, text_features, device='cuda'):
    """
    Evaluates zero-shot classification accuracy on an image dataset.

    Args:
        model (nn.Module): Loaded SigCLIP model for zero-shot evaluation.
        dataloader (DataLoader): DataLoader for evaluation dataset.
        text_features (torch.Tensor): Encoded text embeddings for each class.
        device (str): Device for computation ('cuda' or 'cpu').

    Returns:
        float: Zero-shot classification accuracy as a percentage.
    """
    resize_transform = torch.nn.Sequential(
                             transforms.Resize((224,224)),
                             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))

    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):

            images = batch[0].to(device)
            labels = batch[1].to(device)

            # Reshape images to 224x224
            images = resize_transform(images)

            # Encode images
            image_features = model.extract_image_features(images)
            image_features = F.normalize(image_features, p=2, dim=-1)

            # Compute similarity logits
            logits = image_features @ text_features.t()

            # Predict classes
            predictions = logits.argmax(dim=1)

            # Update accuracy
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    accuracy = (correct / total) * 100
    return accuracy

def zero_shot_classification_pipeline(model, class_names, data_folder='./data', batch_size=128, num_workers=16, device='cuda'):
    """
    Executes the zero-shot classification pipeline from data loading to evaluation.

    Args:
        model (nn.Module): Pre-trained CLIP style model for zero-shot classification.
        class_names (list of str): List of class names for classification.
        data_folder (str): Path to dataset folder.
        batch_size (int, optional): Batch size for DataLoader. Defaults to 128.
        num_workers (int, optional): Number of workers for data loading. Defaults to 16.
        device (str, optional): Device for computations ('cuda' or 'cpu'). Defaults to 'cuda'.

    Returns:
        float: Zero-shot classification accuracy as a percentage.
    """
    # Prepare DataLoader
    dataloader = prepare_dataloader(data_folder, batch_size, num_workers, split='test')

    # Create and encode class prompts
    prompts = create_class_prompts(class_names)
    text_features = encode_prompts(prompts, model, device)

    # Evaluate
    accuracy = evaluate_zero_shot(model, dataloader, text_features, device)
    
    print(f'Zero-Shot Classification Accuracy: {accuracy:.2f}%')
    return accuracy


if __name__ == "__main__":
    model_path = './saved_models/sigclip_baseline.pth'  # Replace with your model path
    class_names = ['airplanes', 'cars', 'birds', 'cats', 'deer', 'dogs', 'frogs', 'horses', 'ships', 'trucks']
    
    image_encoder = ResNet50(include_fc=False)
    text_encoder = TextEncoder(model_name="distilbert-base-uncased", pretrained=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SigCLIP(image_encoder, text_encoder, proj_dim=3).to(device)

    zero_shot_classification_pipeline(
        model=model,
        class_names=class_names,
        batch_size=32,
        num_workers=4,
        device='cuda'
    )