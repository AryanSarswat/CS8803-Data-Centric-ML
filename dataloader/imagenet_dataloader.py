import json
import os

import torch
import torchvision
import torchvision.transforms as transforms
from transformers import AutoProcessor


def get_imagenet_classes(args):
    return [x[1] for x in json.load(open(os.path.join(args.data_folder, "ImageNet/imagenet_class_index.json"))).values()]

def collate_fn(data, prompts, processor):
    images, labels = zip(*data)
    inputs = processor(text=prompts, images=images, padding="max_length", return_tensors="pt")
    return inputs['pixel_values'], inputs['input_ids'], torch.Tensor(labels)

def get_imagenet_dataloader(args, batch_size=4, num_workers=8):
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
	])
    test_set = torchvision.datasets.ImageNet(root=os.path.join(args.data_folder, 'ImageNet'), split='val', transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    # processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
    # test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=lambda data: collate_fn(data, prompts, processor))

    return None, test_loader