import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from collections import defaultdict


def stratified_subset_dataloader(dataloader, fraction=0.1):
    """
    Crée un DataLoader avec un sous-ensemble stratifié du DataLoader d'origine.

    Args:
    - dataloader: DataLoader original.
    - fraction: Fraction du DataLoader à sélectionner (10% par défaut).

    Returns:
    - Un DataLoader contenant un sous-ensemble stratifié.
    """
    # Calculer la taille totale du sous-ensemble
    subset_size = int(fraction * len(dataloader.dataset))

    # Initialiser un dictionnaire pour collecter les indices par classe
    indices_by_class = defaultdict(list)

    # Remplir le dictionnaire avec les indices de chaque classe
    for idx, data in enumerate(dataloader.dataset):
        # inputs = data['input_ids']
        labels = data['label']
        if isinstance(labels, torch.Tensor):
            labels = labels.item()
        indices_by_class[labels].append(idx)

    # Sélectionner un nombre proportionnel d'indices pour chaque classe
    subset_indices = []
    for class_indices in indices_by_class.values():
        class_subset_size = max(1, int(fraction * len(class_indices)))
        selected_indices = np.random.choice(class_indices, size=class_subset_size, replace=False)
        subset_indices.extend(selected_indices)
    print("class_subset_size", class_subset_size)
    # Créer le sous-ensemble
    subset = Subset(dataloader.dataset, subset_indices)
    
    # Créer un nouveau DataLoader pour le sous-ensemble
    subset_dataloader = DataLoader(subset, batch_size=dataloader.batch_size, shuffle=True)

    return subset_dataloader

