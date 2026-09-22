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
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from path_info import PATH

PATH_DATA_CUB = PATH+"/datasets/CUB_200_2011"

class CUBDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        # import dataset file
        self.dataset = pd.read_csv(os.path.join(root, 'cub_concept_dataset_new.csv'))

        # remove concept columns whose sum is 0 (concept never activated)
        self.dataset = self.dataset[self.dataset[[column for column in self.dataset.columns if column.startswith('concept_')]].sum(axis=1) != 0]

        self.dataset['path'] = self.dataset['path'].apply(lambda x: os.path.join(root, 'images', x))

        self.concept_list = self.dataset.columns[self.dataset.columns.str.startswith('concept_')].tolist()
        self.class_dict = {i: c for i, c in enumerate(self.dataset['class_label'].unique())}

        # remove duplicates in concept_list by keeping the same ordering
        self.concept_list = list(dict.fromkeys(self.concept_list))

        print(self.dataset.head(5))

    def __len__(self):
        return len(self.dataset)

    @lru_cache(maxsize=1000)
    def _load_and_transform_image(self, path):
        with Image.open(path) as img:
            if self.transform:
                return self.transform(img)
            return img

    def __getitem__(self, idx):
        image_id = self.dataset.iloc[idx]['image_id']
        image = self._load_and_transform_image(self.dataset.iloc[idx]['path'])
        text_label = self.dataset.iloc[idx]["class_label"]
        label = [k for k, v in self.class_dict.items() if v == text_label][0]
        concept_dict = {concept: self.dataset.iloc[idx][concept] for concept in self.concept_list}

        return {'image_id': image_id ,"image": image, "label": label, **concept_dict}

class ConvertToRGB:
    def __call__(self, image):
        return image.convert('RGB') if image.mode != 'RGB' else image

def logreg(data, class_dict):

    # convert section to integer
    data = data.copy()
    data['target'] = data["class_label"].apply(lambda x: [k for k, v in class_dict.items() if v == x][0])

    # Define features and encoded target
    train_indexes = data['split'] != 'test'
    test_indexes = data['split'] == 'test'
    X = data.drop(columns=['class_label', 'class_id', 'path', 'image_id', 'target', 'split'])
    y = data['target']

    # Train test split
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X.loc[train_indexes]
    y_train = y.loc[train_indexes]
    X_test = X.loc[test_indexes]
    y_test = y.loc[test_indexes]

    # Train classifier
    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    print('Test acc:', accuracy)
    print('Test f1:', f1)

    feature_impact_sum = np.abs(clf.coef_).sum(axis=0)  # Sum across classes of abs coeff
    coeff_dict = dict(zip(X.columns, feature_impact_sum))

    # Calculate mutual information between each feature and target
    mi = mutual_info_classif(X, y, discrete_features='auto', random_state=42)
    mi_dict = dict(zip(X.columns, mi))

    print('coeff_dict')
    print(coeff_dict)
    #print('mi_dict')
    #print(mi_dict)

    return accuracy, f1, coeff_dict, mi_dict

def load_data_CUB(data_dir, batch_size=32, num_workers=4):
    transform = transforms.Compose([
                ConvertToRGB(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )])

    full_dataset = CUBDataset(root=data_dir, transform=transform)
    print('Dataset loaded')

    num_classes = len(full_dataset.dataset['class_label'].unique())
    print(f"Number of classes: {num_classes}")

    print(f"Number of concepts: {len(full_dataset.concept_list)}")

    # create dictionnary to map class index to class names
    class_dict = full_dataset.class_dict

    # concept_list
    concept_list = full_dataset.concept_list

    # make a dict of count % of 1 for each concept column
    concept_counts = {concept: full_dataset.dataset[concept].sum() / len(full_dataset.dataset) for concept in concept_list}
    # for concept in concept_list:
    #     list_count_per_class = []
    #     for label in concepts_N24.keys():
    #         list_count_per_class.append(full_dataset.dataset.loc[full_dataset.dataset['class_label']==label,concept].sum())
        
    #     concept_counts[concept+f'_classactivations'] = np.sum(list_count_per_class)/np.max(list_count_per_class)

    # # calculate vif for each concept column
    # X = add_constant(full_dataset.dataset.drop(columns=['class_label', 'class_id', 'path', 'image_id']))

    # vif_dict = {col: variance_inflation_factor(X.values, i)
    #             for i, col in enumerate(X.columns)
    #             if col != 'const'}
    
    # print('vif_dict')
    # print(vif_dict)

    # Initialize StratifiedKFold
    # skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    # full_labels = full_dataset.dataset['class_label'].values

    # First split: Train and Test/Validation
    # train_indices_raw, test_val_indices_raw = next(skf.split(full_dataset.dataset, full_labels))

    # Second split: Validation and Test (on Test/Validation subset)
    # test_val_labels = full_labels[test_val_indices_raw]
    # val_indices_raw, test_indices_raw = next(skf.split(full_dataset.dataset.iloc[test_val_indices_raw], test_val_labels))

    # Map raw indices back to original dataset indices
    # val_indices = test_val_indices_raw[val_indices_raw]
    # test_indices = test_val_indices_raw[test_indices_raw]

    # combine train and test_indices (no test)
    #train_indices_raw = np.concatenate((train_indices_raw, test_indices))
    #test_indices = []

    # Create subsets
    train_dataset = Subset(full_dataset, full_dataset.dataset.loc[full_dataset.dataset['split']=='train'].index)
    val_dataset = Subset(full_dataset, full_dataset.dataset.loc[full_dataset.dataset['split']=='val'].index)
    test_dataset = Subset(full_dataset, full_dataset.dataset.loc[full_dataset.dataset['split']=='test'].index)

    # Print the number of instances in each dataset
    print(f"Number of instances - Train: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    # # linear regression
    accuracy, f1, coeff_dict, mi_dict = logreg(full_dataset.dataset, class_dict)

    info_dict = {'reg acc': accuracy, 'reg f1': f1, 'coeff_dict': coeff_dict, 'mi_dict': mi_dict}
    #info_dict['vif_dict'] = vif_dict

    # return train_loader, val_loader, test_loader, class_dict, concept_list, concept_counts, info_dict
    return train_loader, val_loader, test_loader, class_dict, concept_list, concept_counts