import pickle
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch

class CocoDataset(Dataset):
    def __init__(self, data_path, category_mapping_path, images_dir, transform=None):
        """
        Args:
            data_path (str): Path to the processed pickle file containing image info.
            category_mapping_path (str): Path to the pickle file containing category to index mapping.
            images_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        with open(data_path, 'rb') as f:
            self.image_info = pickle.load(f)
        
        with open(category_mapping_path, 'rb') as f:
            self.category_to_idx = pickle.load(f)
        
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        file_name, label = self.image_info[idx]
        img_path = os.path.join(self.images_dir, file_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label_idx = self.category_to_idx[label]
        return image, label_idx

def get_coco_dataloader(args):
    train_pickle = os.path.join(args.pickle_folder, 'processed_coco_train.pickle')
    val_pickle = os.path.join(args.pickle_folder, 'processed_coco_val.pickle')
    category_mapping_pickle = os.path.join(args.pickle_folder, 'cat_id_2_label.pickle')
    train_images_dir = os.path.join(args.image_data_folder, 'train2017')
    val_images_dir = os.path.join(args.image_data_folder, 'val2017')

    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Initialize datasets
    train_dataset = CocoDataset(
        data_path=train_pickle,
        category_mapping_path=category_mapping_pickle,
        images_dir=train_images_dir,
        transform=train_transform
    )

    val_dataset = CocoDataset(
        data_path=val_pickle,
        category_mapping_path=category_mapping_pickle,
        images_dir=val_images_dir,
        transform=test_transform
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader, val_loader


def main():
    # Define paths
    train_pickle = 'processed_coco_train.pickle'
    val_pickle = 'processed_coco_val.pickle'
    category_mapping_pickle = 'cat_id_2_label.pickle'
    train_images_dir = '../coco/images/train2017'  
    val_images_dir = '../coco/images/val2017'     

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Initialize datasets
    train_dataset = CocoDataset(
        data_path=train_pickle,
        category_mapping_path=category_mapping_pickle,
        images_dir=train_images_dir,
        transform=transform
    )

    val_dataset = CocoDataset(
        data_path=val_pickle,
        category_mapping_path=category_mapping_pickle,
        images_dir=val_images_dir,
        transform=transform
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)

    # Iterate through the training data
    for images, labels in train_loader:
        print(f'Batch of images shape: {images.shape}')
        print(f'Batch of labels: {labels}')
        break  # Remove this break to iterate through the entire dataset

    # Similarly, iterate through the validation data
    for images, labels in val_loader:
        print(f'Batch of images shape: {images.shape}')
        print(f'Batch of labels: {labels}')
        break  # Remove this break to iterate through the entire dataset

if __name__ == "__main__":
    main()