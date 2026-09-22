import numpy as np

def rank_macro_concepts(score_by_class, only_concept=False):
    """
    Calcule les concepts en moyennant les scores TCAV sur toutes les classes et les trie du plus grand au plus petit.

    Args:
    - score_by_class: Dictionnaire contenant les scores par classe pour chaque concept.
    - only_concept: Si True, retourne uniquement les noms des concepts. Sinon, retourne les noms des concepts et leurs scores moyens.
    
    Returns:
    - Liste des noms des concepts les plus importants, ou liste de tuples (concept, score moyen) si only_concept est False.
    """
    concept_scores = {}

    # Itérer sur chaque classe pour agréger les scores par concept
    for concepts in score_by_class.values():
        for concept, score in concepts.items():
            if concept not in concept_scores:
                concept_scores[concept] = []
            concept_scores[concept].append(score)

    # Calculer la moyenne des scores pour chaque concept
    averaged_scores = {concept: np.mean(scores) for concept, scores in concept_scores.items()}

    # Trier les concepts par score moyen décroissant
    sorted_concepts = sorted(averaged_scores.items(), key=lambda item: item[1], reverse=True)

    # Retourner les résultats en fonction du paramètre only_concept
    if only_concept:
        return [concept for concept, _ in sorted_concepts]
    else:
        return sorted_concepts

def most_k_important_macro_concepts(sorted_concepts_macro_concepts, concept_list, k=1, only_concept=True):
    """
    Prend une liste de concepts et renvoie les k concepts les plus importants basés sur leur rang dans sorted_concepts_macro_concepts.
    
    Args:
    - sorted_concepts_macro_concepts: Liste triée des concepts avec leurs scores moyens (par exemple, sortie de rank_macro_concepts).
    - concept_list: Liste des concepts à évaluer.
    - k: Nombre de concepts à retourner.
    - only_concept: Si True, retourne uniquement les noms des concepts. Sinon, retourne les concepts et leurs scores.
    
    Returns:
    - Les k meilleurs concepts basés sur leur classement dans sorted_concepts_macro_concepts.[list]
    """
    
    # Filtrer sorted_concepts_macro_concepts pour ne conserver que ceux présents dans la liste fournie
    filtered_concepts = [concept for concept in sorted_concepts_macro_concepts if concept[0] in concept_list]
    
    # Vérifier si k est supérieur au nombre de concepts filtrés disponibles
    k = min(k, len(filtered_concepts))
    
    # Retourner les k concepts les plus importants
    if only_concept:
        top_concepts = [concept[0] for concept in filtered_concepts[:k]]
        return top_concepts
    else:
        return filtered_concepts[:k]


import random

def randomize_scores(concepts):
    """
    Assigne un score aléatoire entre 0 et 1 à chaque concept, tout en gérant les formats liste de listes et liste de tuples.

    Args:
    - concepts: Liste contenant des concepts sous la forme [(concept, score), ...] ou [[concept, score], ...].

    Returns:
    - Liste de tuples (concept, nouveau_score), triée du plus grand au plus petit score.
    """

    if not concepts or not isinstance(concepts, list):
        raise ValueError("concepts doit être une liste de tuples (concept, score) ou une liste de listes [[concept, score]].")
    
    # Vérification et normalisation : transformer les listes en tuples pour uniformiser
    normalized_concepts = [(c[0], random.uniform(0, 1)) for c in concepts if isinstance(c, (list, tuple)) and len(c) == 2]

    # Trier par score décroissant
    return sorted(normalized_concepts, key=lambda x: x[1], reverse=True)

def get_concept_at_rank(sorted_concepts, concept_list, i=1, only_concept=True):
    """
    Retourne le concept situé au rang i dans sorted_concepts parmi ceux présents dans concept_list.
    Gère les deux formats : liste de tuples [(concept, score), ...] ou liste de listes [[concept, score], ...].

    Args:
    - sorted_concepts: Liste triée des concepts avec leurs scores [(concept, score), ...] ou [[concept, score], ...].
    - concept_list: Liste des concepts à considérer.
    - i: Rang du concept à récupérer (1 = premier, 2 = deuxième, ...).
    - only_concept: Si True, retourne uniquement le nom du concept, sinon retourne (concept, score).

    Returns:
    - Le concept au rang i ou None s'il n'existe pas. sous format list [] pour faciliter le extend dans self.joint_model.concept_names
    """
    
    if not sorted_concepts or not isinstance(sorted_concepts, list):
        raise ValueError("sorted_concepts doit être une liste de tuples (concept, score) ou une liste de listes [[concept, score]].")
    
    # Vérification et normalisation : transformer les listes en tuples pour homogénéiser le traitement
    if all(isinstance(c, list) and len(c) == 2 for c in sorted_concepts):
        sorted_concepts = [tuple(c) for c in sorted_concepts]  # Convertit [[concept, score], ...] en [(concept, score), ...]

    if not all(isinstance(c, tuple) and len(c) == 2 for c in sorted_concepts):
        raise ValueError("sorted_concepts doit contenir uniquement des paires (concept, score).")
    
    # Filtrer les concepts présents dans concept_list
    filtered_concepts = [concept for concept in sorted_concepts if concept[0] in concept_list]
    
    if not filtered_concepts or i > len(filtered_concepts) or i <= 0:
        return None  # Aucun concept disponible à ce rang

    # Récupérer le concept au rang i-1 (car indexation commence à 0)
    return [filtered_concepts[i-1][0]] if only_concept else filtered_concepts[i-1]



