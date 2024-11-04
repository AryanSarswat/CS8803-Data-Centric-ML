import pickle

import numpy as np
import json
import torch
from torch import nn
from tqdm import tqdm
from collections import Counter

if __name__ == "__main__":
    train_json = "../coco/annotations/instances_train2017.json"
    val_json = "../coco/annotations/instances_val2017.json"

    with open(train_json, 'r') as f:
        train_annotations = json.load(f)
    
    with open(val_json, 'r') as f:
        val_annotations = json.load(f)

    # Preprocess annotations to map image_id to category_ids
    def build_image_to_categories_map(annotations):
        image_to_categories = {}
        for ann in tqdm(annotations['annotations'], desc="Building Annotation Category Mapping"):
            image_id = ann['image_id']
            category_id = ann['category_id']
            if image_id in image_to_categories:
                image_to_categories[image_id].append(category_id)
            else:
                image_to_categories[image_id] = [category_id]
        return image_to_categories

    train_image_to_categories = build_image_to_categories_map(train_annotations)
    val_image_to_categories = build_image_to_categories_map(val_annotations)

    train_image_info = []
    val_image_info = []

    # Process training images
    for img in tqdm(train_annotations['images'], desc="Processing Train Images"):
        image_id = img['id']
        file_name = img['file_name']
        categories = train_image_to_categories.get(image_id, [])
        
        if categories:
            label = categories[0]
            train_image_info.append((file_name, label))

    # Process validation images
    for img in tqdm(val_annotations['images'], desc="Processing Val Images"):
        image_id = img['id']
        file_name = img['file_name']
        categories = val_image_to_categories.get(image_id, [])

        if categories:
            label = categories[0]
            val_image_info.append((file_name, label))
    
    all_labels = []

    for _, label in train_image_info:
        all_labels.append(label)

    for _, label in val_image_info:
        all_labels.append(label)

    category_ids = sorted(set(all_labels))
    category_to_idx = {cat_id: idx for idx, cat_id in enumerate(category_ids)}

    print(f"Length of Train : {len(train_image_info)}")
    print(f"Length of Val : {len(val_image_info)}")
    print(f"Number of Categories : {len(category_ids)}")

    for idx in range(3):
        print(train_image_info[idx])
        print(val_image_info[idx])

    def determine_label_distribution(train_info, val_info, category_mapping):
        """
        Determines and visualizes the distribution of labels in the training and validation datasets.

        Args:
            train_info (list of tuples): List containing (file_name, label) for training data.
            val_info (list of tuples): List containing (file_name, label) for validation data.
            category_mapping (dict): Mapping from category ID to index.
        """
        # Combine labels from both train and val
        all_train_labels = [label for _, label in train_info]
        all_val_labels = [label for _, label in val_info]

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

    # Determine and visualize label distribution
    determine_label_distribution(train_image_info, val_image_info, category_to_idx)

    with open('processed_coco_train.pickle', 'wb') as handle:
        pickle.dump(train_image_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('processed_coco_val.pickle', 'wb') as handle:
        pickle.dump(val_image_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('cat_id_2_label.pickle', 'wb') as handle:
        pickle.dump(category_to_idx, handle, protocol=pickle.HIGHEST_PROTOCOL)