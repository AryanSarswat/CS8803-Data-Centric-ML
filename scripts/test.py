import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloader.cifar_dataloader import get_cifar10_dataloader, get_cifar100_dataloader, get_imagenet_dataloader
from models.sigclip import SigCLIP, sigclip_loss
from models.resnet_vision_encoder import ResNet50
from models.text_encoder import TextEncoder

def load_sigclip_model(model_path, device='cuda'):
    """
    Load the trained SigCLIP model.

    Args:
        model_path (str): Path to the trained SigCLIP weights.
        device (str): Device to load the model on.

    Returns:
        SigCLIP: Loaded SigCLIP model.
    """
    image_encoder = ResNet25(1000, include_fc=False)
    text_encoder = TextEncoder(model_name="distilbert-base-uncased", pretrained=True)

    # Initialize SigCLIP
    sigclip_model = SigCLIP(image_encoder, text_encoder)
    sigclip_model.load(model_path)
    sigclip_model.to(device)
    sigclip_model.eval()
    
    return sigclip_model

def prepare_dataloader(prompts, batch_size=32, num_workers=4, split='test'):
    """
    Prepare the DataLoader for the dataset.

    Args:
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses for data loading.
        split (str): Dataset split to use ('train', 'val', 'test').

    Returns:
        DataLoader: PyTorch DataLoader for the dataset.
    """
    # train_dataloader, test_dataloader = get_cifar10_dataloader(batch_size=batch_size, num_workers=num_workers)
    test_dataloader = get_imagenet_dataloader(prompts, batch_size=batch_size, num_workers=num_workers)

    return test_dataloader

def create_class_prompts(class_names):
    """
    Create textual prompts for each class.

    Args:
        class_names (list): List of class names.

    Returns:
        list: List of formatted prompts.
    """
    return [f"A photo of a {class_name}." for class_name in class_names]

def encode_prompts(prompts, sigclip_model, device='cuda'):
    """
    Encode class prompts into text embeddings.

    Args:
        prompts (list): List of class prompts.
        sigclip_model (SigCLIP): Trained SigCLIP model.
        tokenizer: Tokenizer compatible with the text encoder.
        device (str): Device for computation.

    Returns:
        torch.Tensor: Normalized text feature embeddings.
    """
    encoded = sigclip_model.text_encoder.tokenizer(prompts, padding=True, return_tensors='pt').to(device)
    with torch.no_grad():
        text_features = sigclip_model.extract_text_features(
            encoded['input_ids'],
            encoded['attention_mask']
        )
        text_features = F.normalize(text_features, p=2, dim=-1)
    return text_features

def evaluate_zero_shot(sigclip_model, dataloader, prompts, device='cuda'):
    """
    Evaluate zero-shot classification accuracy.

    Args:
        sigclip_model (SigCLIP): Trained SigCLIP model.
        dataloader (DataLoader): DataLoader for the evaluation dataset.
        prompts (str): class labels
        device (str): Device for computation.

    Returns:
        float: Classification accuracy in percentage.
    """
    resize_transform = transforms.Resize((224, 224))

    correct = 0
    total = 0
    
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            
            images = batch[0].to(device)
            input_ids = batch[1].to(device)
            labels = batch[2].to(device)

            logits = sigclip_model(pixel_values=images, input_ids=input_ids).logits_per_image

            # Predict classes
            predictions = logits.argmax(dim=1)

            # Update accuracy
            all_predictions.append(predictions)
            all_labels.append(labels)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            if i > 100:
                break

    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    both = torch.stack([all_predictions, all_labels], axis=1)
    correct_all = all_predictions == all_labels
    print(both.tolist())
    print(both[correct_all])
    print("ALL ACCURACY", correct_all.sum().item()/all_labels.shape[0])
    accuracy = (correct / total) * 100
    print("accuracy", accuracy)
    1/0    
    return accuracy

def zero_shot_classification_pipeline(sigclip_model, class_names, batch_size=128, num_workers=16, device='cuda'):
    """
    Execute the zero-shot classification pipeline.

    Args:
        model_path (str): Path to the trained SigCLIP model.
        dataset_path (str): Path to the evaluation dataset.
        class_names (list): List of class names for classification.
        batch_size (int, optional): Batch size for DataLoader. Defaults to 32.
        num_workers (int, optional): Number of DataLoader workers. Defaults to 4.
        device (str, optional): Computation device. Defaults to 'cuda'.

    Returns:
        float: Zero-shot classification accuracy.
    """
    # Create and encode class prompts
    prompts = create_class_prompts(class_names)

    # Prepare DataLoader
    dataloader = prepare_dataloader(prompts, batch_size, num_workers, split='test')

    # Evaluate
    accuracy = evaluate_zero_shot(sigclip_model, dataloader, prompts, device)
    
    print(f'Zero-Shot Classification Accuracy: {accuracy:.2f}%')
    return accuracy

if __name__ == "__main__":
    class_names = ['airplanes', 'cars', 'birds', 'cats', 'deer', 'dogs', 'frogs', 'horses', 'ships', 'trucks']
    
    image_encoder = ResNet50(include_fc=False)
    text_encoder = TextEncoder(model_name="distilbert-base-uncased", pretrained=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SigCLIP(image_encoder, text_encoder).to(device)

    zero_shot_classification_pipeline(
        sigclip_model=model,
        class_names=class_names,
        batch_size=32,
        num_workers=4,
        device='cuda'
    )