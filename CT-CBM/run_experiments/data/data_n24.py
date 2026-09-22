import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
import torch.nn.functional as F
from torchvision import transforms, datasets
from transformers import BlipModel, BlipProcessor
from tqdm import tqdm
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold
import numpy as np
import glob
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score, classification_report
import re
from functools import lru_cache
import multiprocessing
import mmap
import json

from data_N24 import PATH_DATA_N24, concepts_N24

class N24NewsDataset(Dataset):
    def __init__(self, root, transform=None, max_len=512, n_instances=None, class_list = None, dataset_type='C3M', combine_type='combine', select_most_frequent=None):
        self.root = root
        self.transform = transform
        self.max_len = max_len
        self.combine_type = combine_type
        self.dataset_type = dataset_type
        self.n_instances = n_instances
        self.class_list = class_list

        if('C3M' in self.dataset_type):
            file_text = f'augmented_dataset_{self.dataset_type.replace("_acc","")}/N24_news_text_aug_{dataset_type}_clean.json'
            file_image = f'augmented_dataset_{self.dataset_type.replace("_acc","")}/N24_news_image_aug_{dataset_type}_clean.json'
        elif(self.dataset_type == 'CBLLM'):
            file_text = f'augmented_dataset_{self.dataset_type}/nytimes_dataset_concepts.json'
            file_image = f'augmented_dataset_{self.dataset_type}/nytimes_dataset_concepts_images.json'

        with open(os.path.join(root, file_text), 'r') as file:
            self.text_dataset = pd.DataFrame(json.load(file))
            if(self.n_instances is not None):
                self.text_dataset = self.text_dataset.sample(n=self.n_instances)
            if(combine_type == 'concat'):
                self.text_dataset.columns = [column.replace('concept_', 'concept_text_') for column in self.text_dataset.columns]

        with open(os.path.join(root, file_image), 'r') as file:
            self.image_dataset = pd.DataFrame(json.load(file))
            if(self.n_instances is not None):
                self.image_dataset = self.image_dataset.sample(n=self.n_instances)
            if(combine_type == 'concat'):
                self.image_dataset.columns = [column.replace('concept_', 'concept_image_') for column in self.image_dataset.columns]

        self.text_dataset.drop(columns=['headline','caption'], inplace=True, errors='ignore')
        self.image_dataset.drop(columns=['headline','caption'], inplace=True, errors='ignore')

        # correct image_id column : split by "/" and only keep last piece
        self.text_dataset['image_id'] = self.text_dataset['image_id'].apply(lambda x: x.split('/')[-1].replace('.jpg', ''))
        self.image_dataset['image_id'] = self.image_dataset['image_id'].apply(lambda x: x.split('/')[-1].replace('.jpg', ''))

        if(self.combine_type == 'combine'):
            # sum concepts columns and put max to 1

            df1_indexed = self.text_dataset.set_index(['image_id', 'section'])
            df2_indexed = self.image_dataset.set_index(['image_id', 'section'])
            df2_indexed = df2_indexed.reindex(df1_indexed.index)

            concept_cols = [col for col in df1_indexed.columns if col.startswith('concept_')]

            self.dataset = df1_indexed.copy()
            self.dataset[concept_cols] = df1_indexed[concept_cols] + df2_indexed[concept_cols]
            if(self.dataset_type != 'CBLLM'):
                self.dataset[concept_cols] = self.dataset[concept_cols].clip(upper=1)

            self.dataset = self.dataset.reset_index()

        elif(self.combine_type == 'concat'):
            display(self.text_dataset)
            display(self.image_dataset)

            if('abstract' in self.image_dataset.columns):
                self.dataset = pd.merge(self.text_dataset, self.image_dataset, on=['abstract','image_id','section'])
            else:
                self.dataset = pd.merge(self.text_dataset, self.image_dataset, on=['image_id','section'])

        self.dataset['image_id'] = self.dataset['image_id'].apply(lambda x: os.path.join(root, 'imgs', x)+'.jpg')
        if(class_list is not None):
            self.dataset = self.dataset[self.dataset['section'].isin(class_list)]

        # remove concept columns whose sum is 0 (concept never activated)
        self.dataset = self.dataset[self.dataset[[column for column in self.dataset.columns if column.startswith('concept_')]].sum(axis=1) != 0]

        self.concept_list = self.dataset.columns[self.dataset.columns.str.startswith('concept_')].tolist()
        self.class_dict = {i: c for i, c in enumerate(self.dataset['section'].unique())}

        # remove duplicates in concept_list by keeping the same ordering
        self.concept_list = list(dict.fromkeys(self.concept_list))

        if(select_most_frequent is not None):
            new_concept_list = []
            for class_name in self.class_dict.values():
                # sum concepts columns along x-axis and sort by largest sum
                ranked_concepts = self.dataset.loc[self.dataset['section']==class_name][self.concept_list].sum(axis=0).sort_values(ascending=False)
                # select only concepts from concepts_N24[class_name]
                ranked_concepts = ranked_concepts[[concept for concept in ranked_concepts.index if any(substring in concept for substring in concepts_N24[class_name])]].sort_values(ascending=False)
                # filter if concept is already in list
                ranked_concepts = ranked_concepts[[concept for concept in ranked_concepts.index if concept not in new_concept_list]].sort_values(ascending=False)
                # save to list
                new_concept_list += ranked_concepts.head(select_most_frequent).index.tolist()

            self.concept_list = new_concept_list

            # select columns from self.concept_list
            self.dataset = self.dataset[['abstract','image_id','section',*self.concept_list]]

        display(self.dataset)

    def __len__(self):
        return len(self.dataset)

    @lru_cache(maxsize=1000)
    def _load_and_transform_image(self, path):
        with Image.open(path) as img:
            if self.transform:
                return self.transform(img)
            return img

    def __getitem__(self, idx):
        image = self._load_and_transform_image(self.dataset.iloc[idx]['image_id'])
        abstract = self.dataset.iloc[idx]["abstract"][:self.max_len].ljust(self.max_len)
        text_label = self.dataset.iloc[idx]["section"]
        label = [k for k, v in self.class_dict.items() if v == text_label][0]
        image_id = self.dataset.iloc[idx]['image_id']
        concept_dict = {concept: self.dataset.iloc[idx][concept] for concept in self.concept_list}

        return {"image": image, "text": abstract, "label": label, "text_label": text_label, 'image_id': image_id, **concept_dict}

class ConvertToRGB:
    def __call__(self, image):
        return image.convert('RGB') if image.mode != 'RGB' else image

def load_data_N24(data_dir, class_list, batch_size=32, num_workers=4, max_len=512, dataset_type='C3M', combine_type='combine', select_most_frequent=None):
    transform = transforms.Compose([
        ConvertToRGB(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    full_dataset = N24NewsDataset(root=data_dir, transform=transform, max_len=max_len, class_list=class_list, dataset_type=dataset_type, combine_type=combine_type, select_most_frequent=select_most_frequent)
    print('Dataset loaded')

    num_classes = len(full_dataset.dataset['section'].unique())
    print(f"Number of classes: {num_classes}")

    print(f"Number of concepts: {len(full_dataset.concept_list)}")

    # create dictionnary to map class index to class names
    class_dict = full_dataset.class_dict

    # concept_list
    concept_list = full_dataset.concept_list

    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    full_labels = full_dataset.dataset['section'].values

    # First split: Train and Test/Validation
    train_indices_raw, test_val_indices_raw = next(skf.split(full_dataset.dataset, full_labels))

    # Second split: Validation and Test (on Test/Validation subset)
    test_val_labels = full_labels[test_val_indices_raw]
    val_indices_raw, test_indices_raw = next(skf.split(full_dataset.dataset.iloc[test_val_indices_raw], test_val_labels))

    # Map raw indices back to original dataset indices
    val_indices = test_val_indices_raw[val_indices_raw]
    test_indices = test_val_indices_raw[test_indices_raw]

    # combine train and test_indices (no test)
    #train_indices_raw = np.concatenate((train_indices_raw, test_indices))
    #test_indices = []

    # Create subsets
    train_dataset = Subset(full_dataset, train_indices_raw)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    # Print the number of instances in each dataset
    print(f"Number of instances - Train: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, class_dict, concept_list