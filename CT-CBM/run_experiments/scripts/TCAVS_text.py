import os 
import sys

import numpy as np
import torch 
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import json

from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.metrics import f1_score, confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from scipy.spatial.distance import euclidean

from torch.utils.data import DataLoader, Subset
from TCAVS_utils import stratified_subset_dataloader

class TCAV():
    """ TCVAS class for ranking concepts importance by assigning a sensitivity score to each concept. """

    def __init__(self, concepts, baseline_model, embedder_tokenizer, batch_size=64, verbose=False, svm_params_dict=None, config = None, train_loader = None, val_loader = None, test_loader = None):
        super().__init__()
        self.embedder_model = baseline_model.embedder_model  # Utiliser le modèle chargé
        self.embedder_tokenizer = embedder_tokenizer  # Utiliser le tokenizer chargé
        self.seed = 42
        self.concepts = concepts
        self.device = config.device
        self.batch_size = batch_size
        self.cavs = {}
        self.verbose = verbose
        # default paramters in sklearn library
        if not svm_params_dict:
            self.svm_params_dict = {
                'alpha': 0.0001,
                'max_iter': 1000,
                'tol': 0.001,
                'class_weight': 'balanced',
            }
        self.score_by_class = {}
        self.sorted_concepts_macro_concepts = None
        self.config = config
        self.use_cls_token = config.use_cls_token
        
        # besoin d'un baseline classifier pour calculer les gradients
        self.baseline_model = baseline_model
        
        # ✅ Détection CLIP/BLIP pour adapter get_pooled_output et get_embeddings_and_labels
        self._is_clip = getattr(baseline_model, '_is_clip', False)
        self._is_clip_or_blip = hasattr(baseline_model, '_is_clip')  # True si BaselineModel_clip_text
        
    def fit(self, dataloader, layer_num = -1, linear_classifier = True, balanced=True, closest=False):
        """ Fit the explainer to the data."""
        self.concept_accuracies = {}
        self.concept_f1score = {}
        for concept in self.concepts:
            cav_, accuracy_, f1score_ = self.learn_cav(dataloader, concept, layer_num, linear_classifier, balanced, closest)
            self.cavs[concept] = cav_
            self.concept_accuracies[concept] = accuracy_
            self.concept_f1score[concept] = f1score_

        file_path = f"{self.config.SAVE_PATH}/blue_checkpoints/{self.config.model_name}/cavs/{self.config.cavs_type}/"
        os.makedirs(f"{file_path}", exist_ok=True)
        with open(file_path+'cavs_svm.json'+ self.config.annotation, 'w') as f:
            json.dump({key: value.tolist() for key, value in self.cavs.items()}, f, indent=4)
        with open(file_path+'cavs_acc.json'+ self.config.annotation, 'w') as f:
            json.dump(self.concept_accuracies, f, indent=4)
        with open(file_path+'cavs_f1.json'+ self.config.annotation, 'w') as f:
            json.dump(self.concept_f1score, f, indent=4)
        print(f"Concepts saved to {file_path}")

        return self.cavs, self.concept_accuracies, self.concept_f1score
         
    def get_embeddings_and_labels(self, dataloader, concept, layer_num=-1, use_cls_token=True):
        """ Get the embedding and the label of the concept.
        
        Handles both standard transformers (BERT) and CLIP/BLIP text-only models.
        For CLIP: uses get_text_features() which returns [B, 512] directly.
        For BLIP: uses text_model() which returns pooler_output [B, 768].
        For BERT: uses standard model(input_ids, attention_mask) with hidden states.
        """
        all_embeddings = []
        all_labels = []
        for batch in dataloader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            with torch.no_grad():
                if self._is_clip_or_blip:
                    # CLIP/BLIP text-only: use dedicated text feature extraction
                    if self._is_clip:
                        # CLIP: get_text_features returns [B, 512] directly
                        pooled_output = self.embedder_model.get_text_features(
                            input_ids=input_ids, 
                            attention_mask=attention_mask
                        )
                    else:
                        # BLIP: text_model returns object with pooler_output [B, 768]
                        outputs = self.embedder_model.text_model(
                            input_ids=input_ids, 
                            attention_mask=attention_mask
                        )
                        pooled_output = outputs.pooler_output
                else:
                    # Standard transformers (BERT, RoBERTa, etc.)
                    outputs = self.embedder_model(
                        input_ids=input_ids, 
                        attention_mask=attention_mask, 
                        output_hidden_states=True
                    )
                    if use_cls_token:
                        if layer_num == -1:
                            pooled_output = outputs.last_hidden_state[:, 0, :]
                        else:
                            pooled_output = outputs.hidden_states[layer_num][:, 0, :]
                    else:
                        if layer_num == -1:
                            pooled_output = outputs.last_hidden_state.mean(1)
                        else:
                            pooled_output = outputs.hidden_states[layer_num].mean(1)

            all_embeddings.append(pooled_output.cpu().numpy())
            all_labels.append(batch[concept].numpy())

        embeddings = np.concatenate(all_embeddings, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        return embeddings, labels
        
    @staticmethod
    def undersample_dataloader(dataloader, concept, balanced=True, closest=False):
        from collections import Counter
        import numpy as np
        from torch.utils.data import DataLoader, Subset

        dataset = dataloader.dataset
        labels = []
        indices = []

        for idx in range(len(dataset)):
            sample = dataset[idx]
            label = sample[concept]
            labels.append(label)
            indices.append(idx)

        labels = np.array(labels)
        indices = np.array(indices)

        class_counts = Counter(labels)
        min_class_count = min(class_counts.values())
        min_class_label = [label for label, count in class_counts.items() if count == min_class_count][0]

        retain_indices = []

        for class_label, count in class_counts.items():
            class_indices = indices[labels == class_label]

            if count == min_class_count:
                retain_indices.extend(class_indices)
            else:
                if closest:
                    embeddings = [dataset[idx_topic]['input_ids'] for idx_topic in indices[labels == min_class_label]]
                    mean_topic_embed = torch.mean(torch.stack(embeddings).float(), dim=0)
                    distances = [euclidean(dataset[idx]['input_ids'], mean_topic_embed) for idx in class_indices]
                    ranked_class_indices = class_indices[np.argsort(distances)]

                if balanced:
                    sampled_indices = np.random.choice(class_indices, size=min_class_count, replace=False)
                    if closest:
                        sampled_indices = ranked_class_indices[:min_class_count]
                else:
                    sampled_indices = np.random.choice(class_indices, size=class_counts[0], replace=False)
                retain_indices.extend(sampled_indices)

        undersampled_dataset = Subset(dataset, retain_indices)
        undersampled_loader = DataLoader(undersampled_dataset, batch_size=dataloader.batch_size, shuffle=True)

        return undersampled_loader


    def learn_cav(self, dataloader, concept, layer_num=-1, linear_classifier=True, 
                  balanced=True, closest=False, logistic_classifier=False, test_dataloader=None):
        """
        Apprend le CAV pour un concept donné.
        """
        concept_idx = self.concepts.index(concept)
    
        original_labels = [batch[concept].numpy() for batch in dataloader]
        original_labels = np.concatenate(original_labels)
        unique, counts = np.unique(original_labels, return_counts=True)
        print(f'Avant undersampling: {dict(zip(unique, counts))}')
    
        undersampled_dataloader = self.undersample_dataloader(dataloader, concept=concept, 
                                                              balanced=balanced, closest=closest)
    
        embeddings, labels = self.get_embeddings_and_labels(undersampled_dataloader, 
                                                              concept=concept, 
                                                              layer_num=layer_num, 
                                                              use_cls_token=self.use_cls_token)
                        
        undersampled_labels = [batch[concept].numpy() for batch in undersampled_dataloader]
        undersampled_labels = np.concatenate(undersampled_labels)
        unique, counts = np.unique(undersampled_labels, return_counts=True)
        print(f'Après undersampling: {dict(zip(unique, counts))}')
    
        if len(set(labels)) > 2:
            raise NotImplementedError('Les CAVs sont définis pour des problèmes binaires.')
    
        if test_dataloader is not None:
            train_embeddings, train_labels = embeddings, labels
            test_embeddings, test_labels = self.get_embeddings_and_labels(test_dataloader, 
                                                                           concept=concept, 
                                                                           layer_num=layer_num, 
                                                                           use_cls_token=self.use_cls_token)
        else:
            train_embeddings, test_embeddings, train_labels, test_labels = \
                train_test_split(embeddings, labels, test_size=0.2, random_state=self.seed, stratify=labels)
    
        unique, counts = np.unique(train_labels, return_counts=True)
        print(f'Train set: {dict(zip(unique, counts))}')
        unique, counts = np.unique(test_labels, return_counts=True)
        print(f'Test set: {dict(zip(unique, counts))}')
    
        if linear_classifier:
            lm = linear_model.SGDClassifier(**self.svm_params_dict)
        elif logistic_classifier:
            lm = linear_model.LogisticRegression()
        else:
            lm = SVC(kernel='rbf', class_weight='balanced')
        
        lm.fit(train_embeddings, train_labels)
        
        train_predictions = lm.predict(train_embeddings)
        test_predictions = lm.predict(test_embeddings)
        
        train_accuracy = accuracy_score(train_labels, train_predictions)
        train_f1score = f1_score(train_labels, train_predictions, average='binary')
        train_conf_matrix = confusion_matrix(train_labels, train_predictions).tolist()
        train_report = classification_report(train_labels, train_predictions, output_dict=True)
        
        test_accuracy = accuracy_score(test_labels, test_predictions)
        test_f1score = f1_score(test_labels, test_predictions, average='binary')
        test_conf_matrix = confusion_matrix(test_labels, test_predictions).tolist()
        test_report = classification_report(test_labels, test_predictions, output_dict=True)
        
        print("Rapport de classification (Test):")
        print(classification_report(test_labels, test_predictions))
        
        performance_dict = {
            "train": {
                "accuracy": train_accuracy,
                "f1score": train_f1score,
                "confusion_matrix": train_conf_matrix,
                "classification_report": train_report
            },
            "test": {
                "accuracy": test_accuracy,
                "f1score": test_f1score,
                "confusion_matrix": test_conf_matrix,
                "classification_report": test_report
            }
        }
        
        performance_file = f"{self.config.SAVE_PATH}/blue_checkpoints/{self.config.model_name}/cavs/{self.config.cavs_type}/performance_{concept}.json"
        os.makedirs(os.path.dirname(performance_file), exist_ok=True)
        with open(performance_file, 'w') as f:
            json.dump(performance_dict, f, indent=4)
        
        if linear_classifier or logistic_classifier:
            cav = -1 * lm.coef_[0]
            cav = cav / np.linalg.norm(cav)
        else:
            print("Impossible d'obtenir les CAVs avec ce type de classifieur, un vecteur nul est retourné.")
            cav = np.zeros(768)
        
        if self.verbose:
            print(f'Learned CAV for concept: {concept}')
            print(f'{list(labels).count(1)} (concept) vs {list(labels).count(0)} (others)')
            print(f'\tTrain Accuracy: {train_accuracy * 100:.1f}% - Test Accuracy: {test_accuracy * 100:.1f}%')
            print(f'\tTrain f1-score: {train_f1score * 100:.1f}% - Test f1-score: {test_f1score * 100:.1f}%')
            print(f'\tTrain confusion matrix: {train_conf_matrix}')
            print(f'\tTest confusion matrix: {test_conf_matrix}')
            print()
    
        return cav, test_accuracy, test_f1score


    def save_cavs_to_file(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.cavs, f)

    def load_cavs_from_file(self, path = None):
        """
        Charge les CAVs depuis un fichier JSON et les stocke dans self.cavs.
        """
        if path == None:
            file_path = f"{self.config.SAVE_PATH}/blue_checkpoints/{self.config.model_name}/cavs/{self.config.cavs_type}/cavs_{self.config.cavs_type}_{self.config.annotation}.json"
            with open(file_path, 'r') as f:
                cavs = json.load(f)
            print("cavs loaded at", file_path)
        else:
            file_path = path
            with open(file_path, 'r') as f:
                cavs = json.load(f)
            print("cavs loaded at", file_path)
        for key, value in cavs.items():
            cavs[key] = np.array(value)
        self.cavs = cavs
        return self.cavs
    
    def get_pooled_output(self, input_ids, attention_mask):
        """
        Get pooled output from the embedder model.
        
        Handles both standard transformers (BERT) and CLIP/BLIP text-only models.
        For CLIP: uses get_text_features() → [B, 512]
        For BLIP: uses text_model() → pooler_output [B, 768]  
        For BERT: uses model(input_ids, attention_mask) → last_hidden_state with CLS/mean pooling
        
        NOTE: This method is called WITHOUT torch.no_grad() because gradients are needed
        for TCAV sensitivity computation.
        """
        if self._is_clip_or_blip:
            # CLIP/BLIP text-only mode
            if self._is_clip:
                # CLIP: get_text_features returns [B, 512] directly (projected)
                pooled_output = self.embedder_model.get_text_features(
                    input_ids=input_ids, 
                    attention_mask=attention_mask
                )
            else:
                # BLIP: text_model returns object with pooler_output [B, 768]
                outputs = self.embedder_model.text_model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask
                )
                pooled_output = outputs.pooler_output
        else:
            # Standard transformers (BERT, RoBERTa, DeBERTa, etc.)
            outputs = self.embedder_model(input_ids=input_ids, attention_mask=attention_mask)
            if self.use_cls_token:
                pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS token
            else:
                pooled_output = outputs.last_hidden_state.mean(1)  # Mean pooling
        
        return pooled_output

    def get_gradients_per_class(self, input_ids, attention_mask, class_idx=0):
        """
        Calcule le gradient par rapport à pooled_output pour une classe donnée.
        """
        classifier = self.baseline_model.classifier
        classifier.eval()
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        pooled_output = self.get_pooled_output(input_ids, attention_mask)
        pooled_output = torch.autograd.Variable(pooled_output, requires_grad=True)
        pooled_output.retain_grad()

        linear_output = classifier(pooled_output)
        target_output = linear_output[:, class_idx]

        classifier.zero_grad()
        target_output.backward()
        
        gradients = pooled_output.grad.clone().detach().cpu().numpy()
        return gradients


    def partition_dataloader_by_labels(self, main_dataloader):
        """ Partitionne un DataLoader en plusieurs DataLoaders en fonction des labels. """
        indices_by_label = {}
        
        for idx, batch in enumerate(main_dataloader):
            labels = batch['label']
            for i, label in enumerate(labels):
                if label.item() not in indices_by_label:
                    indices_by_label[label.item()] = []
                indices_by_label[label.item()].append(idx * main_dataloader.batch_size + i)

        partitioned_loaders = {}
        for label, indices in indices_by_label.items():
            subset = Subset(main_dataloader.dataset, indices)
            partitioned_loaders[label] = DataLoader(subset, batch_size=main_dataloader.batch_size, shuffle=True)

        return partitioned_loaders

    def calculate_tcav_scores_non_linearity_hypothesis_20(self, dataloader=None, max_examples_per_concept=20):
        """
        Calcule les scores TCAV pour chaque concept et chaque classe, en utilisant un maximum
        de max_examples_per_concept exemples par concept.
        """
        import time
        from random import shuffle
        start_time = time.time()
    
        if dataloader is None:
            print("Aucun DataLoader fourni.")
            return None
        num_classes = len(set([d['label'] for d in dataloader.dataset]))
        print(f"Nombre de classes : {num_classes}")
    
        score_by_class = {class_idx: {} for class_idx in range(num_classes)}

        for concept in self.concepts:
            print(f"\n🧠 Concept : {concept}")
            examples = []
    
            for batch in dataloader:
                for i in range(len(batch["input_ids"])):
                    example = {
                        "input_ids": batch["input_ids"][i],
                        "attention_mask": batch["attention_mask"][i]
                    }
                    examples.append(example)
                    if len(examples) >= max_examples_per_concept:
                        break
                if len(examples) >= max_examples_per_concept:
                    break
    
            if len(examples) == 0:
                print(f"Aucun exemple trouvé pour le concept {concept}")
                continue
    
            for class_idx in range(num_classes):
                sensitivity_by_idx = []
    
                for idx, example in enumerate(examples):
                    input_ids = example["input_ids"].unsqueeze(0).to(self.device)
                    attention_mask = example["attention_mask"].unsqueeze(0).to(self.device)
    
                    grads = self.get_gradients_per_class(input_ids, attention_mask, class_idx=class_idx)
                    sensitivity = np.dot(self.cavs[concept], grads.flatten())
                    sensitivity_by_idx.append(sensitivity)
                    print(f"[Classe {class_idx}] Exemple {idx}: sensibilité = {sensitivity:.4f}")
    
                score = len([s for s in sensitivity_by_idx if s > 0]) / len(sensitivity_by_idx)
                score_by_class[class_idx][concept] = score
    
        self.score_by_class = score_by_class
    
        elapsed_time = time.time() - start_time
        print(f"\n✅ TCAV terminé. Temps total : {elapsed_time:.2f} secondes")
        return score_by_class
    
    
    def calculate_tcav_scores(self, dataloader=None, use_subset=True):
        """
        Calcule les scores TCAV pour chaque concept et chaque classe en utilisant un seul exemple par classe.
        """
        score_by_class = {}
        
        if dataloader is None:
            print("Aucun DataLoader fourni.")
            return None
    
        partitioned_loaders = self.partition_dataloader_by_labels(dataloader)
        num_classes = len(partitioned_loaders)
        print(f"Nombre de classes: {num_classes}")
    
        for class_idx in tqdm(range(num_classes), desc="Classes", unit='class'):
            score_by_concept = {}
            
            filtered_dataloader = partitioned_loaders.get(class_idx)
            if filtered_dataloader is None:
                print(f"Aucun exemple pour la classe {class_idx} dans le DataLoader.")
                score_by_class[class_idx] = {}
                continue
    
            if use_subset:
                filtered_dataloader = stratified_subset_dataloader(filtered_dataloader, fraction=0.1)
            
            try:
                batch = next(iter(filtered_dataloader))
            except StopIteration:
                print(f"Aucun exemple dans le DataLoader pour la classe {class_idx}.")
                score_by_class[class_idx] = {}
                continue
            
            input_ids = batch["input_ids"][0].unsqueeze(0).to(self.device)
            attention_mask = batch["attention_mask"][0].unsqueeze(0).to(self.device)
    
            for concept in self.concepts:
                grads = self.get_gradients_per_class(input_ids, attention_mask, class_idx=class_idx)
                sensitivity = np.dot(self.cavs[concept], grads.flatten())
                score = 1 if sensitivity > 0 else 0
                score_by_concept[concept] = score
    
            score_by_class[class_idx] = score_by_concept
    
        self.score_by_class = score_by_class
        
        output_dir = f"{self.config.SAVE_PATH}/blue_checkpoints/{self.config.model_name}/cavs/{self.config.cavs_type}"
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/scores_by_class.pkl", 'wb') as f:
            pickle.dump(score_by_class, f)
    
        return score_by_class

    
    def plot_scores(self, save_path):
        """Trace les scores TCAV par catégorie."""
        categories = list(self.score_by_class[0].keys())
        indices = list(self.score_by_class.keys())
        values = np.array([[self.score_by_class[idx].get(cat, 0) for cat in categories] for idx in indices])

        category_labels = []
        for cat in categories:
            elements = [f"{elem}" for elem in self.score_by_class[0].keys()]
            label = f"{cat} ({', '.join(elements)})"
            category_labels.append(label)

        bar_width = 0.2
        positions = [np.arange(len(indices))]
        for i in range(1, len(categories)):
            positions.append([x + bar_width for x in positions[i - 1]])

        fig, ax = plt.subplots(figsize=(12, 6))

        for i, label in enumerate(category_labels):
            ax.bar(positions[i], values[:, i], width=bar_width, edgecolor='grey', label=label)

        ax.set_xlabel('Index', fontweight='bold')
        ax.set_ylabel('Scores', fontweight='bold')
        ax.set_title('Scores par catégorie et par index')
        ax.set_xticks([r + bar_width for r in range(len(indices))])
        ax.set_xticklabels(indices)
        ax.legend()

        plt.tight_layout()
        plt.savefig(f"{save_path}/blue_checkpoints/{self.config.model_name}/cavs/{self.config.cavs_type}/scores_by_class.png")
        plt.show()

    def save_tcv_ranker(self, file_path):
        """Sauvegarde l'objet TCAV."""
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
        print(f"TCAV object saved to {file_path}")
    
    @staticmethod
    def load_tcv_ranker(file_path):
        """Charge un objet TCAV."""
        with open(file_path, 'rb') as f:
            loaded_tcv_ranker = pickle.load(f)
        print(f"TCAV object loaded from {file_path}")
        return loaded_tcv_ranker