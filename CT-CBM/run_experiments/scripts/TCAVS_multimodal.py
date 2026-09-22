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
    """ TCVAS class for ranking concepts importance by assigning a sensitivity score to each concept. 
    
    Args:
    - concepts: List of concepts to explain.
    - device: Device to use for computations.
    - batch_size: Batch size for DataLoader.
    - verbose: If True, print additional information.
    - svm_params_dict: Dictionary containing parameters for the SVM classifier.
    - num_classes: Number of classes in the dataset.
    
    Attributes:
    - model: The model to use for explanations.
    - tokenizer: The tokenizer to use for explanations.
    - max_len: Maximum length of input sequences.
    - seed: Random seed for reproducibility.
    - concepts: List of concepts to explain.
    - batch_size: Batch size for DataLoader.
    - cavs: Dictionary containing the concept-attribute vectors.
    - verbose: If True, print additional information.
    
    Methods:
    - fit: Fit the explainer to the data.
    - get_gradient: Get the gradient of the model with respect to the embedding.
    - get_embeddings: Get the embeddings of the data.
    - learn_cav: Learn the concept-attribute vector for a given concept.
    - save_cavs_to_file: Save the concept-attribute vectors to a file.
    - load_cavs_from_file: Load the concept-attribute vectors from a file.
    - get_gradients: Get the gradients of the model with respect to the embedding.
    - get_gradients_per_class: Get the gradients of the model with respect to the embedding for a given class.
    - calculate_tcav_scores: Calculate the TCAV scores for each concept and class.
    - most_important_concepts: Calculate the most important concepts.
    - plot_scores: Plot the TCAV scores
    """

    def __init__(self, concepts, baseline_model, embedder_tokenizer = None, batch_size=64, verbose=False, svm_params_dict=None, config = None, train_loader = None, val_loader = None, test_loader = None):
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
        
    def fit(self, dataloader, layer_num = -1, linear_classifier = True, balanced=True, closest=False):
        """ Fit the explainer to the data.
        
            Args:
            - dataloader: DataLoader containing the data.
            - classifier: The (SVM) classifier to use for explanations .            
            Returns:
            - Update the cav vector for each concept."""
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
        """ get the embedding and the label of the concept to avoid dataloader sampling strategy to disrupt everything"""
        all_embeddings = []
        all_labels = []
        for batch in dataloader:    
            images = batch['image'].to(self.device)
            texts = batch['text']

            inputs = self.baseline_model.tokenizer(text=texts, images=images, return_tensors="pt", padding=True, truncation=True)
            # déplacer chaque champ sur le device
            for k, v in inputs.items():
                inputs[k] = v.to(self.device)

            with torch.no_grad():
                for concept in self.concepts:
                    pooled_output = self.baseline_model.get_pooled_output(inputs.input_ids, inputs.attention_mask, inputs.pixel_values, class_idx=class_idx)
                
       
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

        # Accéder au dataset depuis le DataLoader
        dataset = dataloader.dataset

        # Initialiser les listes pour stocker les labels et les indices correspondants
        labels = []
        indices = []

        # Parcourir le dataset pour extraire les labels et les indices
        for idx in range(len(dataset)):
            sample = dataset[idx]
            label = sample[concept]
            labels.append(label)
            indices.append(idx)

        labels = np.array(labels)
        indices = np.array(indices)

        # Compter le nombre d'instances pour chaque classe
        class_counts = Counter(labels)

        # Trouver la taille de la classe minoritaire
        min_class_count = min(class_counts.values())

        # Trouver le label correspondant à la classe minoritaire
        min_class_label = [label for label, count in class_counts.items() if count == min_class_count][0]

        # Initialiser la liste des indices à conserver
        retain_indices = []

        # Appliquer l'undersampling
        for class_label, count in class_counts.items():
            class_indices = indices[labels == class_label]

            if count == min_class_count:
                # Garder tous les indices de la classe minoritaire
                retain_indices.extend(class_indices)

            else:

                if(closest):
                    #distances = [sum([euclidean(dataset[idx]['input_ids'], dataset[idx_topic]['input_ids']) for idx_topic in indices[labels == min_class_label]]) for idx in class_indices]
                    embeddings = [dataset[idx_topic]['input_ids'] for idx_topic in indices[labels == min_class_label]]
                    mean_topic_embed = torch.mean(torch.stack(embeddings).float(), dim=0)
                    distances = [euclidean(dataset[idx]['input_ids'], mean_topic_embed) for idx in class_indices]
                    ranked_class_indices = class_indices[np.argsort(distances)]

                # Sélectionner aléatoirement un nombre égal d'indices dans la classe majoritaire
                if(balanced):
                    sampled_indices = np.random.choice(class_indices, size=min_class_count, replace=False)
                    if(closest):
                        sampled_indices = ranked_class_indices[:min_class_count]
                else:
                    sampled_indices = np.random.choice(class_indices, size=class_counts[0], replace=False)
                retain_indices.extend(sampled_indices)

        # Créer un nouveau DataLoader en utilisant les indices sous-échantillonnés
        undersampled_dataset = Subset(dataset, retain_indices)
        undersampled_loader = DataLoader(undersampled_dataset, batch_size=dataloader.batch_size, shuffle=True)

        return undersampled_loader



    def learn_cav(self, dataloader, concept, layer_num=-1, linear_classifier=True, 
                  balanced=True, closest=False, logistic_classifier=False, test_dataloader=None):
        """
        Apprend le CAV pour un concept donné en entraînant un classifieur sur les embeddings extraits
        d'un dataloader (après undersampling) et évalue sa performance sur un dataloader de test (si fourni).
        Les performances (train et test) sont sauvegardées dans un fichier JSON.
        
        Args:
            dataloader (DataLoader): DataLoader pour l'entraînement.
            concept (str): Nom du concept pour lequel apprendre le CAV.
            layer_num (int): Indice de la couche à utiliser pour extraire les embeddings.
            linear_classifier (bool): Si True, utilise SGDClassifier.
            balanced (bool): Paramètre pour l'undersampling.
            closest (bool): Option pour l'undersampling basé sur la distance.
            logistic_classifier (bool): Si True, utilise LogisticRegression.
            test_dataloader (DataLoader, optionnel): DataLoader pour évaluer les performances du classifieur.
            
        Returns:
            cav (np.array): Le vecteur CAV appris.
            accuracy (float): Précision du classifieur sur l'ensemble de test.
            f1score (float): f1-score du classifieur sur l'ensemble de test.
        """
        # Trouver l'indice du concept actuel dans la liste des concepts (si nécessaire)
        concept_idx = self.concepts.index(concept)
    
        # Compter les éléments avant undersampling
        original_labels = [batch[concept].numpy() for batch in dataloader]
        original_labels = np.concatenate(original_labels)
        unique, counts = np.unique(original_labels, return_counts=True)
        print(f'Avant undersampling: {dict(zip(unique, counts))}')
    
        # Appliquer l'undersampling sur le dataloader d'entraînement
        undersampled_dataloader = self.undersample_dataloader(dataloader, concept=concept, 
                                                              balanced=balanced, closest=closest)
    
        # Extraire les embeddings et les labels après undersampling pour l'entraînement
        embeddings, labels = self.get_embeddings_and_labels(undersampled_dataloader, 
                                                              concept=concept, 
                                                              layer_num=layer_num, 
                                                              use_cls_token=self.use_cls_token)
                        
        # Compter les éléments après undersampling
        undersampled_labels = [batch[concept].numpy() for batch in undersampled_dataloader]
        undersampled_labels = np.concatenate(undersampled_labels)
        unique, counts = np.unique(undersampled_labels, return_counts=True)
        print(f'Après undersampling: {dict(zip(unique, counts))}')
    
        # Vérifier que le problème est binaire
        if len(set(labels)) > 2:
            raise NotImplementedError('Les CAVs sont définis pour des problèmes binaires.')
    
        # Récupérer les ensembles d'entraînement et de test
        if test_dataloader is not None:
            # Utiliser le dataloader de test fourni pour l'évaluation
            train_embeddings, train_labels = embeddings, labels
            test_embeddings, test_labels = self.get_embeddings_and_labels(test_dataloader, 
                                                                           concept=concept, 
                                                                           layer_num=layer_num, 
                                                                           use_cls_token=self.use_cls_token)
        else:
            # Fractionner aléatoirement l'ensemble d'entraînement en train/test
            train_embeddings, test_embeddings, train_labels, test_labels = \
                train_test_split(embeddings, labels, test_size=0.2, random_state=self.seed, stratify=labels)
    
        # Affichage de la répartition dans les ensembles de train et test
        unique, counts = np.unique(train_labels, return_counts=True)
        print(f'Train set: {dict(zip(unique, counts))}')
        unique, counts = np.unique(test_labels, return_counts=True)
        print(f'Test set: {dict(zip(unique, counts))}')
    
        # Choix du classifieur
        if linear_classifier:
            lm = linear_model.SGDClassifier(**self.svm_params_dict)
        elif logistic_classifier:
            lm = linear_model.LogisticRegression()
        else:
            lm = SVC(kernel='rbf', class_weight='balanced')
        
        # Entraînement du classifieur
        lm.fit(train_embeddings, train_labels)
        
        # Prédictions sur l'ensemble d'entraînement et de test
        train_predictions = lm.predict(train_embeddings)
        test_predictions = lm.predict(test_embeddings)
        
        # Calcul des métriques pour l'ensemble d'entraînement
        train_accuracy = accuracy_score(train_labels, train_predictions)
        train_f1score = f1_score(train_labels, train_predictions, average='binary')
        train_conf_matrix = confusion_matrix(train_labels, train_predictions).tolist()
        train_report = classification_report(train_labels, train_predictions, output_dict=True)
        
        # Calcul des métriques pour l'ensemble de test
        test_accuracy = accuracy_score(test_labels, test_predictions)
        test_f1score = f1_score(test_labels, test_predictions, average='binary')
        test_conf_matrix = confusion_matrix(test_labels, test_predictions).tolist()
        test_report = classification_report(test_labels, test_predictions, output_dict=True)
        
        # Affichage du rapport de classification pour le test
        print("Rapport de classification (Test):")
        print(classification_report(test_labels, test_predictions))
        
        # Sauvegarde des performances dans un dictionnaire
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
        
        # Sauvegarde dans un fichier JSON avec le maximum de détails
        performance_file = f"{self.config.SAVE_PATH}/blue_checkpoints/{self.config.model_name}/cavs/{self.config.cavs_type}/performance_{concept}.json"
        os.makedirs(os.path.dirname(performance_file), exist_ok=True)
        with open(performance_file, 'w') as f:
            json.dump(performance_dict, f, indent=4)
        
        # Extraction du CAV selon le type de classifieur utilisé
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
        Retourne le dictionnaire des CAVs.
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
        # Si nécessaire, convertir les listes en tableaux numpy
        for key, value in cavs.items():
            cavs[key] = np.array(value)
        
        self.cavs = cavs
        return self.cavs
    

    def get_gradients_per_class(self, input_ids, attention_mask, pixel_values, class_idx=0):
        """
        Calcule le gradient par rapport à `final_output` pour une classe donnée pour une seule entrée.

        Args:
        - classifier: Le modèle à utiliser pour le calcul du gradient.
        - input_ids: Les IDs d'entrée de la phrase (une seule entrée).
        - attention_mask: Le masque d'attention de la phrase (une seule entrée).
        - class_idx: L'indice de la classe pour laquelle calculer le gradient.

        Returns:
        - gradients: Les gradients calculés par rapport à `final_output`.
        """
        classifier = self.baseline_model.classifier
        classifier.eval()  # Mettre le modèle en mode évaluation
        input_ids = input_ids.to(self.device)  # Déplacer les inputs sur le bon device
        attention_mask = attention_mask.to(self.device)

        pooled_output = self.baseline_model.get_pooled_output(input_ids, attention_mask, pixel_values)
        pooled_output = torch.autograd.Variable(pooled_output, requires_grad=True)
        pooled_output.retain_grad()

        linear_output = classifier(pooled_output)
        target_output = linear_output[:, class_idx]  # Obtenir la sortie pour la classe spécifiée

        classifier.zero_grad()

        target_output.backward()
        
        gradients = pooled_output.grad.clone().detach().cpu().numpy()
        # print("Weights:", classifier[0].weight[0])
        # print("Gradients:", pooled_output.grad)

        return gradients


    # @staticmethod
    def partition_dataloader_by_labels(self, main_dataloader):
        """ Partitionne un DataLoader en plusieurs DataLoaders en fonction des labels des échantillons.
        Args:
            main_dataloader (DataLoader): Le DataLoader principal à partitionner.
        
        Returns:
            dict: Un dictionnaire de DataLoaders partitionnés par label.
        """
        # Récupérer les indices des échantillons pour chaque classe
        indices_by_label = {}
        
        for idx, batch in enumerate(main_dataloader):
            # On suppose que batch est un dictionnaire contenant 'label'
            labels = batch['label']
            
            for i, label in enumerate(labels):
                if label not in indices_by_label:
                    indices_by_label[label] = []
                indices_by_label[label].append(idx * main_dataloader.batch_size + i)

        # Créer un dictionnaire pour stocker les DataLoaders partitionnés
        partitioned_loaders = {}

        for label, indices in indices_by_label.items():
            subset = Subset(main_dataloader.dataset, indices)
            
            # Créer un DataLoader pour le sous-ensemble
            partitioned_loaders[label] = DataLoader(subset, batch_size=main_dataloader.batch_size, shuffle=True)

        return partitioned_loaders


    def calculate_tcav_scores_non_linearity_hypothesis_20(self, dataloader=None, max_examples_per_concept=20):
        """
        Calcule les scores TCAV pour chaque concept et chaque classe, en utilisant un maximum
        de 20 exemples par concept (indépendamment des classes).
    
        Args:
        - dataloader: DataLoader contenant les données de test.
        - max_examples_per_concept: nombre maximal d'exemples à utiliser par concept.
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
    
        # Pour chaque concept, sélectionner 20 exemples
        for concept in self.concepts:
            print(f"\n Concept : {concept}")
            examples = []
    
            # Parcourir le dataloader jusqu'à avoir 20 exemples
            for batch in dataloader:
                for i in range(len(batch["image"])):
                    example = {
                        "image": batch["image"][i],
                        "text": batch["text"][i]
                    }
                    examples.append(example)
                    if len(examples) >= max_examples_per_concept:
                        break
                if len(examples) >= max_examples_per_concept:
                    break
    
            if len(examples) == 0:
                print(f"Aucun exemple trouvé pour le concept {concept}")
                continue
    
            # Pour chaque classe, calculer les gradients et les scores TCAV
            for class_idx in range(num_classes):
                sensitivity_by_idx = []
    
                for idx, example in enumerate(examples):
                    # input_ids = example["input_ids"].unsqueeze(0).to(self.device)
                    # attention_mask = example["attention_mask"].unsqueeze(0).to(self.device)

                    # label = example['label'].to(self.device)
                    images = example['image'].to(self.device)
                    texts = example['text']

                    inputs = self.baseline_model.tokenizer(text=texts, images=images, return_tensors="pt", padding=True, truncation=True)
                    # déplacer chaque champ sur le device
                    for k, v in inputs.items():
                        inputs[k] = v.to(self.device)
                                            
                    # Obtenir les gradients pour cette classe
                    grads = self.get_gradients_per_class(inputs.input_ids, inputs.attention_mask, inputs.pixel_values, class_idx=class_idx)
    
                    # Calculer la sensibilité (projection du gradient sur le CAV)
                    sensitivity = np.dot(self.cavs[concept], grads.flatten())
                    sensitivity_by_idx.append(sensitivity)
                    print(f"[Classe {class_idx}] Exemple {idx}: sensibilité = {sensitivity:.4f}")
    
                # Calculer le score TCAV
                score = len([s for s in sensitivity_by_idx if s > 0]) / len(sensitivity_by_idx)
                score_by_class[class_idx][concept] = score
    
        self.score_by_class = score_by_class
    
        elapsed_time = time.time() - start_time
        print(f"\n TCAV terminé. Temps total : {elapsed_time:.2f} secondes")
        return score_by_class
    
    
    def calculate_tcav_scores(self, dataloader=None, use_subset=True):
        """
        Calcule les scores TCAV pour chaque concept et chaque classe en utilisant un seul exemple par classe.
        
        Args:
            - dataloader: DataLoader contenant les données de test.
            - use_subset: Si True, utiliser un sous-ensemble stratifié (10%).
            
        Retourne:
            - score_by_class: Dictionnaire contenant les scores TCAV par classe.
        """
        score_by_class = {}
        
        if dataloader is None:
            print("Aucun DataLoader fourni.")
            return None
    
        # Partitionner le DataLoader par labels
        partitioned_loaders = self.partition_dataloader_by_labels(dataloader)
        num_classes = len(partitioned_loaders)
        print(f"Nombre de classes: {num_classes}")
    
        # Pour chaque classe, utiliser un seul exemple pour le calcul
        for class_idx in tqdm(range(num_classes), desc="Classes", unit='class'):
            score_by_concept = {}
            
            # Obtenir le DataLoader filtré pour la classe actuelle
            filtered_dataloader = partitioned_loaders.get(class_idx)
            if filtered_dataloader is None:
                print(f"Aucun exemple pour la classe {class_idx} dans le DataLoader.")
                score_by_class[class_idx] = {}
                continue
    
            # Optionnel : utiliser un sous-ensemble stratifié (ici, 10% des données)
            if use_subset:
                filtered_dataloader = stratified_subset_dataloader(filtered_dataloader, fraction=0.1)
            
            # Extraire un seul exemple de la classe (la première ligne disponible)
            try:
                batch = next(iter(filtered_dataloader))
            except StopIteration:
                print(f"Aucun exemple dans le DataLoader pour la classe {class_idx}.")
                score_by_class[class_idx] = {}
                continue
            
            # On prend le premier exemple du batch
            # input_ids = batch["input_ids"][0].unsqueeze(0).to(self.device)
            # attention_mask = batch["attention_mask"][0].unsqueeze(0).to(self.device)
            images = batch['image'].to(self.device)
            texts = batch['text']

            inputs = self.baseline_model.tokenizer(text=texts, images=images, return_tensors="pt", padding=True, truncation=True)
            # déplacer chaque champ sur le device
            for k, v in inputs.items():
                inputs[k] = v.to(self.device)
                
            # Pour chaque concept, calculer le gradient et la sensibilité sur ce seul exemple
            for concept in self.concepts:
                # Calculer les gradients pour l'exemple individuel
                grads = self.get_gradients_per_class(inputs.input_ids, inputs.attention_mask, inputs.pixel_values, class_idx=class_idx)
                
                # Calculer la sensibilité pour cet exemple
                sensitivity = np.dot(self.cavs[concept], grads.flatten())
                
                # Le score TCAV devient 1 si la sensibilité est positive, sinon 0
                score = 1 if sensitivity > 0 else 0
                score_by_concept[concept] = score
    
            score_by_class[class_idx] = score_by_concept
    
        self.score_by_class = score_by_class
        
        # Enregistrer les résultats
        output_dir = f"{self.config.SAVE_PATH}/blue_checkpoints/{self.config.model_name}/cavs/{self.config.cavs_type}"
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/scores_by_class.pkl", 'wb') as f:
            pickle.dump(score_by_class, f)
    
        return score_by_class

    
    def plot_scores(self, save_path):
        """
        Trace les scores TCAV par catégorie et les sauvegarde sous forme d'image.

        Args:
        - score_by_class: Dictionnaire des scores TCAV par classe.
        - save_path: Chemin pour sauvegarder le graphique.
        - model_name: Le nom du modèle.
        """
        # Extraire les catégories et les valeurs
        categories = list(self.score_by_class[0].keys())
        indices = list(self.score_by_class.keys())
        values = np.array([[self.score_by_class[idx].get(cat, 0) for cat in categories] for idx in indices])

        # Créer des labels personnalisés pour chaque catégorie en fonction des éléments trouvés
        category_labels = []
        for cat in categories:
            elements = [f"{elem}" for elem in self.score_by_class[0].keys()]
            label = f"{cat} ({', '.join(elements)})"
            category_labels.append(label)

        # Définir la largeur des barres
        bar_width = 0.2
        # Calculer les positions des barres pour chaque catégorie
        positions = [np.arange(len(indices))]
        for i in range(1, len(categories)):
            positions.append([x + bar_width for x in positions[i - 1]])

        # Créer le graphique en barres non empilées
        fig, ax = plt.subplots(figsize=(12, 6))

        for i, label in enumerate(category_labels):
            ax.bar(positions[i], values[:, i], width=bar_width, edgecolor='grey', label=label)

        # Ajouter des labels, des titres et des légendes
        ax.set_xlabel('Index', fontweight='bold')
        ax.set_ylabel('Scores', fontweight='bold')
        ax.set_title('Scores par catégorie et par index')
        ax.set_xticks([r + bar_width for r in range(len(indices))])
        ax.set_xticklabels(indices)
        ax.legend()

        # Afficher et sauvegarder le graphique
        plt.tight_layout()
        plt.savefig(f"{save_path}/blue_checkpoints/{config.model_name}/cavs/{config.cavs_type}/scores_by_class.png")
        plt.show()

    def save_tcv_ranker(self, file_path):
        """
        Sauvegarde l'objet TCAV avec tous ses attributs dans un fichier pickle.

        Args:
        - file_path: Chemin du fichier où l'objet sera sauvegardé.
        """
        # Sauvegarder l'objet entier dans un fichier
        # os.makedirs(f"{file_path}", exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
        print(f"TCAV object saved to {file_path}")
    
    @staticmethod
    def load_tcv_ranker(file_path):
        """
        Charge un objet TCAV sauvegardé à partir d'un fichier pickle.

        Args:
        - file_path: Chemin du fichier d'où l'objet sera chargé.
        
        Returns:
        - L'objet TCAV chargé.
        """
        # Charger l'objet depuis un fichier
        with open(file_path, 'rb') as f:
            loaded_tcv_ranker = pickle.load(f)
        print(f"TCAV object loaded from {file_path}")
        return loaded_tcv_ranker