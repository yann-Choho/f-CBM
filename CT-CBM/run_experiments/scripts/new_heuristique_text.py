import pickle, os, gc
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import json
import matplotlib.pyplot as plt

################################################################################################

################################################################################################

#trouvé dans le code source dez CB_LLM
def cos_sim_cubed(cbl_features, target):
    cbl_features = cbl_features - torch.mean(cbl_features, dim=-1, keepdim=True)
    target = target - torch.mean(target, dim=-1, keepdim=True)

    cbl_features = F.normalize(cbl_features**3, dim=-1)
    target = F.normalize(target**3, dim=-1)

    sim = torch.sum(cbl_features*target, dim=-1)
    return sim.mean()
    
# version 1 : f1 score
def compute_cosine_matrix_and_metrics(df, text_column, model, embedder_tokenizer, cavs, 
                                      f1_cutoff=None, device=torch.device("cuda"),
                                      save_dir=None, config=None, annotation = None, cos_cubed = False):
    """
    Pour chaque texte dans df[text_column], calcule l'embedding CLS via embedder_model
    et détermine la similarité cosinus entre cet embedding et chacun des vecteurs CAV.
    
    Le résultat est stocké dans un DataFrame (cosine_df) contenant les scores cosinus pour chaque concept,
    auquel sont ajoutées les colonnes "text" et "label" issues de df.
    
    Ensuite, à partir d'un seuil (threshold), des prédictions sont définies (1 si score > threshold)
    et, en comparant aux colonnes de ground truth dans df, les métriques suivantes sont calculées pour chaque concept :
      - accuracy
      - positive_rate (taux de prédiction positive)
      - F1 score
      
    Les colonnes ground truth dans df peuvent être soit nommées exactement comme le concept,
    soit préfixées par "concept_".
    
    Si save_dir est fourni, la fonction charge cosine_df depuis le fichier de sauvegarde s'il existe,
    sinon elle le calcule et le sauvegarde.
    
    Args:
        df (pd.DataFrame): Doit contenir au moins la colonne text_column, "label", et pour chaque concept,
                           une colonne ground truth portant soit le nom exact du concept, soit "concept_" + concept.
        text_column (str): Nom de la colonne contenant les textes.
        embedder_model (torch.nn.Module): Modèle d'embedder (ex. BERT).
        embedder_tokenizer: Tokenizer associé au modèle.
        cavs (dict): Dictionnaire des vecteurs CAV pour chaque concept.
        threshold (float): Seuil pour définir la prédiction (1 si score > threshold).
        f1_cutoff (float, optional): Seuil pour filtrer les concepts en fonction du F1 score.
        device (torch.device): Le device (ex: torch.device("cuda")).
        save_dir (str, optional): Répertoire pour sauvegarder cosine_df. S'il existe déjà, il sera chargé.
        
    Returns:
        tuple: (cosine_df, metrics, filtered_concepts)
            - cosine_df : DataFrame avec les scores de similarité pour chaque concept, ainsi que "text" et "label".
            - metrics : dictionnaire avec les métriques (accuracy, positive_rate, F1, TP, FP, FN, TN) par concept.
            - filtered_concepts : liste des concepts filtrés en fonction du F1 score (si f1_cutoff est défini), sinon liste de tous les concepts.
    """
    
    import pickle, os, gc
    from sklearn.model_selection import train_test_split

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        cosine_path = os.path.join(save_dir, f"cosine_df_{annotation}.pkl")
    else:
        cosine_path = None

    # Calcul ou chargement de cosine_df
    if cosine_path and os.path.exists(cosine_path):
        print("Chargement de cosine_df depuis", cosine_path)
        with open(cosine_path, "rb") as f:
            cosine_df = pickle.load(f)
    else:
        print("Calcul de cosine_df...")
        model.to(device)
        model.eval()
        cosine_scores = []
        index_list = []

        for idx, row in df.iterrows():
            text = row[text_column]
            encoded_input = embedder_tokenizer(text, return_tensors='pt', truncation=True, padding=True,  return_token_type_ids=False)
            for key in encoded_input:
                encoded_input[key] = encoded_input[key].to(device)
            with torch.no_grad():
                cls_emb = model.get_pooled_output(**encoded_input)


            sample_scores = {}
            for concept, cav_vec in cavs.items():
                if not isinstance(cav_vec, torch.Tensor):
                    cav_vec = torch.tensor(cav_vec, dtype=torch.float32)
                cav_vec = cav_vec.to(device)
                if cav_vec.dim() == 1:
                    cav_vec = cav_vec.unsqueeze(0)
                if cos_cubed :                
                    # Utilisation de la fonction custom pour la "cos cubed similarity"
                    sim = cos_sim_cubed(cav_vec, cls_emb).item()
                else:
                    sim = F.cosine_similarity(cav_vec, cls_emb, dim=1).item()

                sample_scores[concept] = sim
            cosine_scores.append(sample_scores)
            index_list.append(idx)
            del encoded_input, cls_emb
            torch.cuda.empty_cache()

        cosine_df = pd.DataFrame(cosine_scores, index=index_list)
        cosine_df = df[['text', 'label']].join(cosine_df)

        if cosine_path:
            with open(cosine_path, "wb") as f:
                pickle.dump(cosine_df, f)
            print("cosine_df sauvegardé à", cosine_path)

    # Split en train (70%) et val (30%) — mêmes index que dans df
    train_df, val_df = train_test_split(df, test_size=0.3, random_state=42)
    cosine_train = cosine_df.loc[train_df.index]
    cosine_val = cosine_df.loc[val_df.index]

    pred_columns = [col for col in cosine_df.columns if col not in ["text", "label"]]

    # Seuils = médianes sur les données de train
    thresholds = {concept: cosine_train[concept].median() for concept in pred_columns}

    # Génération des prédictions sur val
    predictions = pd.DataFrame(index=cosine_val.index)
    for concept in pred_columns:
        predictions[concept] = (cosine_val[concept] > thresholds[concept]).astype(int)

    # Calcul des métriques sur val
    metrics = {}
    n_samples = len(val_df)
    for concept in cavs.keys():
        if concept in df.columns:
            truth = df.loc[val_df.index, concept]
        elif f"concept_{concept}" in df.columns:
            truth = df.loc[val_df.index, f"concept_{concept}"]
        else:
            print(f"Attention : aucune colonne ground truth pour le concept '{concept}'. On utilise des zéros.")
            truth = pd.Series([0] * n_samples, index=val_df.index)

        pred = predictions[concept]
        TP = ((pred == 1) & (truth == 1)).sum()
        FP = ((pred == 1) & (truth == 0)).sum()
        FN = ((pred == 0) & (truth == 1)).sum()
        TN = ((pred == 0) & (truth == 0)).sum()

        accuracy = (TP + TN) / n_samples if n_samples > 0 else 0.0
        positive_rate = pred.mean()
        f1 = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0.0

        metrics[concept] = {
            "accuracy": accuracy,
            "positive_rate": positive_rate,
            "F1": f1,
            "TP": int(TP),
            "FP": int(FP),
            "FN": int(FN),
            "TN": int(TN)
        }

    # save it
    with open(os.path.join(save_dir,
                           f"detection_concept_{config.cavs_type}_{config.annotation}_{config.agg_mode}_{config.agg_scope}.json"), 'w') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

    
    if f1_cutoff is not None:
        filtered_concepts = [concept for concept, m in metrics.items() if m["F1"] >= f1_cutoff]
    else:
        filtered_concepts = list(cavs.keys())

    filtered_concepts = [clean_concept_name(name) for name in filtered_concepts]

    return cosine_train, cosine_val, cosine_df, thresholds, metrics, filtered_concepts

# version 1 : f1 score avec gemma model
def compute_cosine_matrix_and_metrics_gemma_version(df, text_column, embedder_model, embedder_tokenizer, cavs, 
                                      f1_cutoff=None, device=torch.device("cuda"),
                                      save_dir=None, config=None, annotation = None, cos_cubed = False):
    import pickle, os, gc
    from sklearn.model_selection import train_test_split

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        cosine_path = os.path.join(save_dir, f"cosine_df_{annotation}.pkl")
    else:
        cosine_path = None

    # Calcul ou chargement de cosine_df
    if cosine_path and os.path.exists(cosine_path):
        print("Chargement de cosine_df depuis", cosine_path)
        with open(cosine_path, "rb") as f:
            cosine_df = pickle.load(f)
    else:
        print("Calcul de cosine_df...")
        embedder_model.to(device)
        embedder_model.eval()
        cosine_scores = []
        index_list = []

        for idx, row in df.iterrows():
            text = row[text_column]
            encoded_input = embedder_tokenizer(text,
                                               return_tensors='pt', 
                                               truncation=True, 
                                               padding=True, 
                                               max_length=512, 
                                               return_token_type_ids=False)
            for key in encoded_input:
                encoded_input[key] = encoded_input[key].to(device)
            with torch.no_grad():
                # Passe à travers le modèle d'embedding                
                output = embedder_model(input_ids=encoded_input['input_ids']) #.flatten() pas de ça comme dans la sortie du prepare_data_agnews
                cls_emb = output[0][:,-1,:]  # CLS token
                
            sample_scores = {}
            for concept, cav_vec in cavs.items():
                if not isinstance(cav_vec, torch.Tensor):
                    cav_vec = torch.tensor(cav_vec, dtype=torch.float32)
                cav_vec = cav_vec.to(device)
                if cav_vec.dim() == 1:
                    cav_vec = cav_vec.unsqueeze(0)
                if cos_cubed :                
                    # Utilisation de la fonction custom pour la "cos cubed similarity"
                    sim = cos_sim_cubed(cav_vec, cls_emb).item()
                else:
                    sim = F.cosine_similarity(cav_vec, cls_emb, dim=1).item()

                sample_scores[concept] = sim
            cosine_scores.append(sample_scores)
            index_list.append(idx)
            del encoded_input, cls_emb, output
            torch.cuda.empty_cache()

        cosine_df = pd.DataFrame(cosine_scores, index=index_list)
        cosine_df = df[['text', 'label']].join(cosine_df)

        if cosine_path:
            with open(cosine_path, "wb") as f:
                pickle.dump(cosine_df, f)
            print("cosine_df sauvegardé à", cosine_path)

    # Split en train (70%) et val (30%) — mêmes index que dans df
    train_df, val_df = train_test_split(df, test_size=0.3, random_state=42)
    cosine_train = cosine_df.loc[train_df.index]
    cosine_val = cosine_df.loc[val_df.index]

    pred_columns = [col for col in cosine_df.columns if col not in ["text", "label"]]

    # Seuils = médianes sur les données de train
    thresholds = {concept: cosine_train[concept].median() for concept in pred_columns}

    # Génération des prédictions sur val
    predictions = pd.DataFrame(index=cosine_val.index)
    for concept in pred_columns:
        predictions[concept] = (cosine_val[concept] > thresholds[concept]).astype(int)

    # Calcul des métriques sur val
    metrics = {}
    n_samples = len(val_df)
    for concept in cavs.keys():
        if concept in df.columns:
            truth = df.loc[val_df.index, concept]
        elif f"concept_{concept}" in df.columns:
            truth = df.loc[val_df.index, f"concept_{concept}"]
        else:
            print(f"Attention : aucune colonne ground truth pour le concept '{concept}'. On utilise des zéros.")
            truth = pd.Series([0] * n_samples, index=val_df.index)

        pred = predictions[concept]
        TP = ((pred == 1) & (truth == 1)).sum()
        FP = ((pred == 1) & (truth == 0)).sum()
        FN = ((pred == 0) & (truth == 1)).sum()
        TN = ((pred == 0) & (truth == 0)).sum()

        accuracy = (TP + TN) / n_samples if n_samples > 0 else 0.0
        positive_rate = pred.mean()
        f1 = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0.0

        metrics[concept] = {
            "accuracy": accuracy,
            "positive_rate": positive_rate,
            "F1": f1,
            "TP": int(TP),
            "FP": int(FP),
            "FN": int(FN),
            "TN": int(TN)
        }

    # save it
    with open(os.path.join(save_dir,
                           f"detection_concept_{config.cavs_type}_{config.annotation}_{config.agg_mode}_{config.agg_scope}.json"), 'w') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

    
    if f1_cutoff is not None:
        filtered_concepts = [concept for concept, m in metrics.items() if m["F1"] >= f1_cutoff]
    else:
        filtered_concepts = list(cavs.keys())

    filtered_concepts = [clean_concept_name(name) for name in filtered_concepts]

    return cosine_train, cosine_val, cosine_df, thresholds, metrics, filtered_concepts


################################################################################################
# Step 2 :  selecting by coverage 
################################################################################################

import os
import pickle
import re
import pandas as pd

def clean_concept_name(name):
    """
    Nettoie un nom de concept en supprimant le préfixe "cos_" ou "concept_", en retirant les espaces superflus
    et en supprimant les phrases inutiles à partir de "Let me know...".
    """
    import re
    name = name.replace("cos_", "").replace("concept_", "").strip()
    name = re.sub(r"Let me know.*", "", name)  # Supprime les phrases inutiles
    return " ".join(name.split())  # Réduit les espaces multiples

def rename_dataframe_columns(df):
    """
    Renomme les colonnes du DataFrame en nettoyant les noms contenant les préfixes "cos_" ou "concept_".
    """
    new_columns = {col: clean_concept_name(col) if col.startswith(("cos_", "concept_")) else col for col in df.columns}
    return df.rename(columns=new_columns)

# filter_and_order_concepts
def filter_concepts_by_coverage(df, sorted_concepts, coverage_threshold=0.8, save_path=None):
    """
    Filtre les concepts qui sont bien détectés et les ordonne en fonction de leur importance.

    - Calcule la couverture cumulative en ajoutant successivement les concepts de la liste triée.
    - Retient uniquement les concepts appartenant à `filtered_concepts`.
    - Garde uniquement le minimum de concepts nécessaires pour atteindre le `coverage_threshold`.

    Args:
        df (pd.DataFrame): DataFrame contenant les colonnes de ground truth pour les concepts.
                           On suppose que les colonnes ground truth sont nommées "concept_" + concept.
        sorted_concepts (list): Liste triée de tuples (nom du concept brut, score).
        coverage_threshold (float): Seuil de couverture à atteindre (par défaut 0.8 pour 80%).
        save_path (str, optional): Chemin pour sauvegarder les résultats intermédiaires (pickle).

    Returns:
        tuple: (selected_concepts, cumulative_coverages)
            - selected_concepts : liste minimale des concepts nettoyés permettant d'atteindre au moins le seuil.
            - cumulative_coverages : liste des couvertures cumulatives pour chaque concept ajouté.
    """
    df = rename_dataframe_columns(df)  # Nettoyer les colonnes du DataFrame
    total = len(df)
    print(df.columns)

    # Nettoyer les concepts triés et les filtrer selon ceux bien détectés
    cleaned_sorted_concepts = [clean_concept_name(name)for name in sorted_concepts]

    cumulative_columns = []   # Liste des colonnes à inclure progressivement
    cumulative_coverages = []
    selected_columns = None

    for concept in cleaned_sorted_concepts:
        print(concept)
        col_name = f"concept_{concept}" if f"concept_{concept}" in df.columns else concept
        cumulative_columns.append(col_name)
        
        # Filtrer les lignes où au moins une des colonnes cumulées vaut 1
        filtered_data = df[cumulative_columns][df[cumulative_columns].eq(1).any(axis=1)]
        coverage = len(filtered_data) / total
        cumulative_coverages.append(coverage)

    sorted_macro_concepts_coverage = [(c,v) for c,v in zip(cleaned_sorted_concepts, cumulative_coverages)]
        
    with open(save_path, 'w') as f:
        json.dump(sorted_macro_concepts_coverage, f, ensure_ascii=False, indent=4)

    return sorted_macro_concepts_coverage

################################################################################################
#### s'assurer que le cumul des concepts se fera du plus important concept au moins important
################################################################################################

def order_well_detected_concepts(sorted_concepts, filtered_concepts):
    """
    Filtre la liste triée des concepts pour ne conserver que ceux qui ont été détectés (filtered_concepts)
    tout en préservant l'ordre du plus important au moins important.
    
    Args:
        sorted_concepts (list): Liste de tuples (concept_name, score) triée par ordre décroissant (du plus important au moins important).
                                Par exemple : [("famous people", 0.0393),
                                              ("Athletes and related entities", 0.0347),
                                              ...]
        filtered_concepts (list): Liste des concepts bien détectés (noms nettoyés, sans préfixe "cos_").
    
    Returns:
        list: Liste de tuples (concept, score) pour les concepts filtrés dans le même ordre.
    """
        
    # Filtrer la liste triée pour ne conserver que les concepts présents dans filtered_set
    filtered_sorted = [ (name, score) for name, score in sorted_concepts if name in filtered_concepts ]
    
    return filtered_sorted


#### VIZUALISATION ##############

def plot_concept_threshold(cosine_df, groundtruth_df, concept, thresholds, gt_prefix="concept_"):
    """
    Affiche un scatter plot pour un concept donné :
      - L'axe des y représente les scores de similarité cosinus pour le concept (dans cosine_df).
      - Les points sont colorés en fonction de la ground truth issue de groundtruth_df :
          * Vert si la valeur de ground truth (colonne "concept_<concept>" ou <concept>) vaut 1.
          * Rouge sinon.
      - Une ligne horizontale indique le seuil choisi.
    
    Args:
        cosine_df (pd.DataFrame): DataFrame contenant les scores de similarité pour chaque concept,
                                  avec une colonne 'text' et 'label' qui seront utilisées pour la jointure.
        groundtruth_df (pd.DataFrame): DataFrame contenant la ground truth pour chaque concept,
                                       avec les colonnes 'text' et 'label' pour faire la jointure.
        concept (str): Nom nettoyé du concept (sans "cos_" ou "concept_") pour lequel visualiser.
        thresholds (dict): Dictionnaire contenant les seuils pour chaque concept, par exemple { 'concept1': 0.35, ... }.
        gt_prefix (str): Préfixe attendu dans groundtruth_df pour les colonnes ground truth (par défaut "concept_").
    
    Returns:
        None. Affiche le graphique.
    """
    groundtruth_df.columns = ['concept_' + col if col not in ['text', 'label'] else col for col in groundtruth_df.columns]

    # Effectuer une jointure sur les colonnes 'text' et 'label'
    merged_df = cosine_df.merge(groundtruth_df, on=['text', 'label'], how='inner')
    
    # Déterminer le nom de la colonne ground truth dans le DataFrame fusionné
    gt_col = f"{gt_prefix}{concept}"
    if gt_col not in merged_df.columns:
        if concept in merged_df.columns:
            gt_col = concept
        else:
            raise ValueError(f"Colonne ground truth pour le concept '{concept}' non trouvée dans le DataFrame fusionné.")
    
    # Vérifier que la colonne de similarité est présente
    if concept not in merged_df.columns:
        raise ValueError(f"Colonne de similarité pour le concept '{concept}' non trouvée dans le DataFrame fusionné.")
    
    # Récupérer les scores et la ground truth depuis le DataFrame fusionné
    sims = merged_df[concept]
    truth = merged_df[gt_col]
    
    # Définir la couleur : vert pour ground truth = 1, rouge pour ground truth = 0
    colors = truth.map({1: "green", 0: "red"})
    
    plt.figure(figsize=(10, 6))
    # Utiliser l'index (numéro d'échantillon) comme axe x et les scores de similarité comme axe y
    plt.scatter(range(len(sims)), sims, c=colors, alpha=0.7, edgecolor='k', s=10)
    # Ligne horizontale indiquant le seuil pour le concept
    plt.axhline(y=thresholds[concept], color="blue", linestyle="--", 
                label=f"Threshold = {thresholds[concept]}")
    plt.xlabel("Échantillon")
    plt.ylabel("Cosine Similarity")
    plt.title(f"Cosine Similarity pour le concept '{concept}'")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()
