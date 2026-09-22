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

PATH_DATA_agnews = PATH+"/datasets/agnews"

class agnewsDataset(Dataset):
    def __init__(self, train_val_test='train', dataset_type='C3M', select_concepts=None, random_concepts=None):

        self.dataset_type = dataset_type

        # import dataset file
        if(train_val_test=='train'):
            self.dataset = pd.read_csv(os.path.join(PATH_DATA_agnews, f'df_with_topics_v4_{dataset_type.replace("CBLLM","CB_LLM")}.csv'))
        else:
            self.dataset = pd.read_csv(os.path.join(PATH_DATA_agnews, f'df_with_topics_v4_{dataset_type.replace("CBLLM","CB_LLM")}_test.csv'))

        columns = [c for c in self.dataset.columns if "dummy" not in c and c not in ["Unnamed: 0.1", "Unnamed: 0", "test",'topics','extracted_topics','macro_concepts','state']]

        self.dataset = self.dataset[columns]

        self.dataset.columns = ['concept_text_'+col.lower().replace(' ','_') if (col!='text') and (col!='label') else col for col in self.dataset.columns]

        self.concept_list = [c for c in self.dataset.columns if c.startswith('concept_text_')]

        # only keep in dataset the concept columns contained in the list select_concepts
        if(select_concepts is not None):
            prefix = 'concept_text_'
            concepts_to_drop = [concept for concept in self.concept_list if concept not in [prefix+conc for conc in select_concepts]]
            self.dataset.drop(concepts_to_drop, axis=1, inplace=True)
            self.concept_list = self.dataset.columns[self.dataset.columns.str.startswith('concept_')].tolist()
            self.concept_list = list(dict.fromkeys(self.concept_list)) # unique

        # randomly select concepts
        if(random_concepts is not None):
            self.concept_list = np.random.choice(self.concept_list, random_concepts, replace=False)
            non_concept_cols = [column for column in self.dataset.columns if not column.startswith('concept_')]
            self.dataset = self.dataset[[*non_concept_cols,*self.concept_list]]

        # same order of columns
        non_concept_cols = [column for column in self.dataset.columns if not column.startswith('concept_')]
        self.dataset = self.dataset[non_concept_cols + sorted(self.concept_list)]

        # remove concept columns whose sum is 0 (concept never activated)
        #self.dataset = self.dataset[self.dataset[[column for column in self.dataset.columns if column.startswith('concept_text_')]].sum(axis=1) != 0]

        self.class_dict = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}

        print(self.dataset.head(5))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset.iloc[idx]["text"]
        label = self.dataset.iloc[idx]["label"]
        concept_dict = {concept: self.dataset.iloc[idx][concept] for concept in self.concept_list}

        return {'text':text, "label": label, **concept_dict}

def logreg(data, class_dict):

    print('Predicting final task using true concepts')

    # convert section to integer
    data = data.copy()

    # Define features and encoded target
    X = data.drop(columns=['label','text'])
    y = data['label']

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

def load_data_agnews(batch_size=32, num_workers=4, dataset_type='C3M', combine_type='text', select_concepts=None, random_concepts=None):

    if(combine_type != 'text'):
        raise ValueError("No combine_type other than text is supported")

    # Create subsets
    train_dataset = agnewsDataset(train_val_test='train', dataset_type=dataset_type, select_concepts=select_concepts, random_concepts=random_concepts)

    # concept_list
    concept_list = train_dataset.concept_list

    # select same concepts as train dataset
    val_dataset = agnewsDataset(train_val_test='val', dataset_type=dataset_type, select_concepts=[concept.replace('concept_text_', '') for concept in concept_list])
    test_dataset = agnewsDataset(train_val_test='test', dataset_type=dataset_type, select_concepts=[concept.replace('concept_text_', '') for concept in concept_list])
    print('Dataset loaded')

    num_classes = len(train_dataset.dataset['label'].unique())
    print(f"Number of classes: {num_classes}")

    print(f"Number of concepts: {len(train_dataset.concept_list)}")

    # create dictionnary to map class index to class names
    class_dict = train_dataset.class_dict

    # make a dict of count % of 1 for each concept column
    concept_counts = {concept: train_dataset.dataset[concept].sum() / len(train_dataset.dataset) for concept in concept_list}

    # Print the number of instances in each dataset
    print(f"Number of instances - Train: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    # export datasets
    # check first if file exists or not
    if(select_concepts is None):
        data_dir = PATH_DATA_agnews
        if not os.path.exists(os.path.join(data_dir, f'{combine_type}/{dataset_type.replace("CBLLM","cb_llm")}_annotation/train_df_{dataset_type.replace("CBLLM","cb_llm")}.csv')):
            print('exporting train, val and test datasets')

            train_dataset.dataset.rename(columns={col:col.replace('concept_text_','concept_') for col in train_dataset.dataset.columns if col.startswith('concept_')})[['text','label']+[column.replace('concept_text_','concept_') for column in train_dataset.dataset.columns if column.startswith('concept_')]].to_csv(os.path.join(data_dir, f'{combine_type}/{dataset_type.replace("CBLLM","cb_llm")}_annotation/train_df_{dataset_type.replace("CBLLM","cb_llm")}.csv'), index=False)
            val_dataset.dataset.rename(columns={col:col.replace('concept_text_','concept_') for col in val_dataset.dataset.columns if col.startswith('concept_')})[['text','label']+[column.replace('concept_text_','concept_') for column in val_dataset.dataset.columns if column.startswith('concept_')]].to_csv(os.path.join(data_dir, f'{combine_type}/{dataset_type.replace("CBLLM","cb_llm")}_annotation/val_df_{dataset_type.replace("CBLLM","cb_llm")}.csv'), index=False)
            test_dataset.dataset.rename(columns={col:col.replace('concept_text_','concept_') for col in test_dataset.dataset.columns if col.startswith('concept_')})[['text','label']+[column.replace('concept_text_','concept_') for column in test_dataset.dataset.columns if column.startswith('concept_')]].to_csv(os.path.join(data_dir, f'{combine_type}/{dataset_type.replace("CBLLM","cb_llm")}_annotation/test_df_{dataset_type.replace("CBLLM","cb_llm")}.csv'), index=False)
            with open(os.path.join(data_dir, f'{combine_type}/{dataset_type.replace("CBLLM","cb_llm")}_annotation/label_dict_{dataset_type.replace("CBLLM","cb_llm")}.json'), 'w') as f:
                json.dump(dict(zip(class_dict.values(), class_dict.keys())), f, indent=4)


    # # linear regression
    accuracy, f1, coeff_dict, mi_dict = logreg(train_dataset.dataset, class_dict)

    info_dict = {'reg acc': accuracy, 'reg f1': f1, 'coeff_dict': coeff_dict, 'mi_dict': mi_dict}
    #info_dict['vif_dict'] = vif_dict

    # return train_loader, val_loader, test_loader, class_dict, concept_list, concept_counts, info_dict
    return train_loader, val_loader, test_loader, class_dict, concept_list, concept_counts
