import os
import pickle
import argparse
import numpy as np
import json
import torch
from torch import nn
from tqdm import tqdm
from collections import Counter

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_data_folder', default='../coco', type=str)
    parser.add_argument('--pickle_folder', default='./', type=str)
    args = parser.parse_args()
    return args

class Preprocess:
    def __init__(self, args):
        self.args = args

    # Preprocess annotations to map image_id to category_ids
    def build_image_to_categories_map(self, annotations):
        image_to_categories = {}
        for ann in tqdm(annotations['annotations'], desc="Building Annotation Category Mapping"):
            image_id = ann['image_id']
            category_id = ann['category_id']
            if image_id in image_to_categories:
                image_to_categories[image_id].append(category_id)
            else:
                image_to_categories[image_id] = [category_id]
        return image_to_categories

    def get_annotations(self, split):
        json_file = os.path.join(args.coco_data_folder, f"annotations/instances_{split}2017.json")
        with open(json_file, 'r') as f:
            annotations = json.load(f)
        return annotations

    def get_image_info(self, split):
        annotations = self.get_annotations(split)
        image_to_categories = self.build_image_to_categories_map(annotations)
        image_info = []
        for img in tqdm(annotations['images'], desc=f"Processing {split} Images"):
            image_id = img['id']
            file_name = img['file_name']
            categories = image_to_categories.get(image_id, [])
            if categories:
                label = list(sorted(set(categories))) #[0]
                self.labels[split].extend(label)
                image_info.append((file_name, label))
        return image_info

    def determine_label_distribution(self, train_info, val_info, category_mapping, all_train_labels, all_val_labels):
        """
        Determines and visualizes the distribution of labels in the training and validation datasets.

        Args:
            train_info (list of tuples): List containing (file_name, label) for training data.
            val_info (list of tuples): List containing (file_name, label) for validation data.
            category_mapping (dict): Mapping from category ID to index.
        """
        # Count occurrences
        train_counter = Counter(all_train_labels)
        val_counter = Counter(all_val_labels)

        # Sort categories for consistent plotting
        categories = sorted(category_mapping.keys())

        train_counts = [train_counter[cat] for cat in categories]
        val_counts = [val_counter[cat] for cat in categories]

        # Print the distribution
        print("Label Distribution in Training Set:")
        for cat in categories:
            print(f"Category {cat}: {train_counter[cat]}")

        print("\nLabel Distribution in Validation Set:")
        for cat in categories:
            print(f"Category {cat}: {val_counter[cat]}")

    def save_pickles(self, pickle_folder, train_image_info, val_image_info, category_to_idx):
        with open(os.path.join(pickle_folder, 'processed_coco_train.pickle'), 'wb') as handle:
            pickle.dump(train_image_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(pickle_folder, 'processed_coco_val.pickle'), 'wb') as handle:
            pickle.dump(val_image_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(pickle_folder, 'cat_id_2_label.pickle'), 'wb') as handle:
            pickle.dump(category_to_idx, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def run(self):
        self.labels = {'train':[], 'val':[]}
        train_image_info = self.get_image_info('train')
        val_image_info = self.get_image_info('val')
        all_labels = self.labels['train']+self.labels['val']
        category_ids = sorted(set(all_labels))
        category_to_idx = {cat_id: idx for idx, cat_id in enumerate(category_ids)}

        print(f"Length of Train : {len(train_image_info)}")
        print(f"Length of Val : {len(val_image_info)}")
        print(f"Number of Categories : {len(category_ids)}")

        for idx in range(3):
            print(train_image_info[idx])
            print(val_image_info[idx])
        # Determine and visualize label distribution
        self.determine_label_distribution(train_image_info, val_image_info, category_to_idx, self.labels['train'], self.labels['val'])
        self.save_pickles(self.args.pickle_folder, train_image_info, val_image_info, category_to_idx)


if __name__ == "__main__":
    args = parse_args()
    preprocessor = Preprocess(args)
    preprocessor.run()

    

    

    