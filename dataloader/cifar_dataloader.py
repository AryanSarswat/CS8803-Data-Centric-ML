import torch
import torchvision
import torchvision.transforms as transforms
from transformers import AutoProcessor



def collate_fn(data, prompts, processor):
    images, labels = zip(*data)
    inputs = processor(text=prompts, images=images, padding="max_length", return_tensors="pt")
    return inputs['pixel_values'], inputs['input_ids'], torch.Tensor(labels)

def get_imagenet_dataloader(prompts, batch_size=4, num_workers=8):
    test_set = torchvision.datasets.ImageNet(root='./data', split='val')
    processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=lambda data: collate_fn(data, prompts, processor))

    return test_loader

def get_cifar10_dataloader(batch_size=4, num_workers=8):
    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
         ])
    
    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
         ])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader

def get_cifar100_dataloader(batch_size=4, num_workers=2):
    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
         ])
    
    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
         ])

    train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    test_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader

def get_SHVN_dataloader(batch_size=4, num_workers=4):
    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
         ])
    
    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
         ])

    train_set = torchvision.datasets.SVHN(root='./data', split="train", download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    test_set = torchvision.datasets.SVHN(root='./data', split="test", download=True, transform=transform_test)
    
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader

if __name__ == "__main__":
    # CIFAR10
    train_loader, test_loader = get_cifar10_dataloader()
    print(f" Size of CIFAR10 Train set {len(train_loader)}")
    print(f" Size of CIFAR10 Test set {len(test_loader)}")
    images, labels = next(iter(train_loader))
    print(f"Shape of CIFAR10 Train Images {images.size()}")
    print(f"Shape of CIFAR10 Train Labels {labels.size()}")
    images, labels = next(iter(test_loader))
    print(f"Shape of CIFAR10 Train Images {images.size()}")
    print(f"Shape of CIFAR10 Train Labels {labels.size()}")
    
    
    # CIFAR100
    train_loader, test_loader = get_cifar100_dataloader()
    print(f" Size of CIFAR100 Train set {len(train_loader)}")
    print(f" Size of CIFAR100 Test set {len(test_loader)}")
    images, labels = next(iter(train_loader))
    print(f"Shape of CIFAR100 of Train Images {images.size()}")
    print(f"Shape of CIFAR100 Train Labels {labels.size()}")
    images, labels = next(iter(test_loader))
    print(f"Shape of CIFAR10 Train Images {images.size()}")
    print(f"Shape of CIFAR10 Train Labels {labels.size()}")

    # SHVN
    train_loader, test_loader = get_SHVN_dataloader()
    print(f" Size of SHVN Train set {len(train_loader)}")
    print(f" Size of SHVN Test set {len(test_loader)}")
    images, labels = next(iter(train_loader))
    print(f"Shape of SHVN of Train Images {images.size()}")
    print(f"Shape of SHVN Train Labels {labels.size()}")
    images, labels = next(iter(test_loader))
    print(f"Shape of SHVN Train Images {images.size()}")
    print(f"Shape of SHVN Train Labels {labels.size()}")
    print(labels)