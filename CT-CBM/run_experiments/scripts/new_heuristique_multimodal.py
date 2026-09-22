import pickle, os, gc
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import json
import matplotlib.pyplot as plt


#trouvé dans le code source dez CB_LLM
def cos_sim_cubed(cbl_features, target):
    cbl_features = cbl_features - torch.mean(cbl_features, dim=-1, keepdim=True)
    target = target - torch.mean(target, dim=-1, keepdim=True)

    cbl_features = F.normalize(cbl_features**3, dim=-1)
    target = F.normalize(target**3, dim=-1)

    sim = torch.sum(cbl_features*target, dim=-1)
    return sim.mean()
    


# # version  : 70/30 train/val
def compute_cosine_matrix_and_metrics_multimodal_dataloader(
    dataloader,
    model,
    tokenizer,
    cavs,
    f1_cutoff=None,
    device=torch.device("cuda"),
    save_dir=None,
    config=None,
    annotation=None,
    cos_cubed=False
):
    """
    Itère sur un DataLoader multimodal et calcule les similarités cosinus entre embeddings et vecteurs CAV,
    puis génère un split train/val 70/30 stratifié, définit des seuils via la médiane sur le train,
    et calcule les métriques (accuracy, positive_rate, F1, TP, FP, FN, TN) sur l'ensemble de validation,
    en utilisant la vérité terrain par concept fournie dans le batch.

    Args:
        dataloader: torch.utils.data.DataLoader fournissant les clés :
            - pour chaque concept `c`, batch["concept_{c}"] (vérité terrain 0/1)
            - input_ids, attention_mask, pixel_values
        model: modèle multimodal avec méthode get_pooled_output
        cavs (dict): vecteurs CAV par concept
        f1_cutoff (float, optional): seuil minimal de F1 pour filtrer les concepts
        device: device pour le calcul
        save_dir: répertoire de sauvegarde
        config: config pour noms de fichiers de métriques
        annotation: suffixe pour nom de fichier pickle
        cos_cubed (bool): utiliser la similarité cosinus cubée

    Returns:
        train_df, val_df, cosine_df, thresholds, metrics, filtered_concepts
    """
    import os, pickle, json, gc, torch
    import torch.nn.functional as F
    from sklearn.model_selection import train_test_split
    import pandas as pd

    # Prépare save paths
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        cosine_path = os.path.join(save_dir, f"cosine_df_{annotation}.pkl")
    else:
        cosine_path = None

    # Chargement ou calcul du DataFrame de similarités
    if cosine_path and os.path.exists(cosine_path):
        with open(cosine_path, 'rb') as f:
            cosine_df = pickle.load(f)
    else:
        model.to(device).eval()
        records = []
        # On suppose que chaque batch contient pour chaque concept une clé 'concept_{concept}'
        for batch in tqdm(dataloader, desc="cosine per batch", unit="batch"):            
            texts = batch['text']
            images = batch['image'].to(device)

            inputs = tokenizer(
                text=texts,
                images=images,
                return_tensors='pt',
                padding=True,
                truncation=True,
                #do_rescale=False
            )
            for k, v in inputs.items():
                inputs[k] = v.to(device)

            with torch.no_grad():
                cls_emb = model.get_pooled_output(
                    inputs.input_ids,
                    inputs.attention_mask,
                    inputs.pixel_values
                )  # [B, D]

            for idx, emb in enumerate(cls_emb):
                sample = {'text': texts[idx]}
                # Calcul similarité et ajout vérité terrain par concept
                for concept, cav_vec in cavs.items():
                    # vérité terrain pour ce concept
                    gt = int(batch[f'concept_{concept}'][idx].item())
                    sample[f'concept_{concept}'] = gt
                    # similarité
                    vec = cav_vec if isinstance(cav_vec, torch.Tensor) else torch.tensor(cav_vec, dtype=torch.float32)
                    vec = vec.to(device)
                    if vec.dim() == 1:
                        vec = vec.unsqueeze(0)
                    if cos_cubed:
                        sim = cos_sim_cubed(vec, emb.unsqueeze(0)).item()
                    else:
                        sim = F.cosine_similarity(vec, emb.unsqueeze(0), dim=1).item()
                    sample[f'sim_{concept}'] = sim
                records.append(sample)
        cosine_df = pd.DataFrame(records)
        if cosine_path:
            with open(cosine_path, 'wb') as f:
                pickle.dump(cosine_df, f)

    # Colonnes de similarités et vérité terrain
    sim_cols = [c for c in cosine_df.columns if c.startswith('sim_')]
    gt_cols = [c for c in cosine_df.columns if c.startswith('concept_')]

    # Split 70/30 stratifié sur la somme des labels si multi-label
    # ici on stratifie sur la présence d'au moins un concept positif
    strat = (cosine_df[gt_cols].sum(axis=1) > 0).astype(int)
    train_df, val_df = train_test_split(
        cosine_df,
        test_size=0.3,
        random_state=42,
        stratify=strat
    )

    # Seuils par concept = médiane sur train
    thresholds = {c: train_df[c].median() for c in sim_cols}

    # Prédictions sur validation
    preds = val_df[sim_cols].gt(pd.Series(thresholds)).astype(int)

    # Calcul des métriques par concept
    metrics = {}
    n_val = len(val_df)
    for sim_c in sim_cols:
        concept = sim_c.replace('sim_', '')
        truth = val_df[f'concept_{concept}']
        pred = preds[sim_c]
        TP = int(((pred == 1) & (truth == 1)).sum())
        FP = int(((pred == 1) & (truth == 0)).sum())
        FN = int(((pred == 0) & (truth == 1)).sum())
        TN = int(((pred == 0) & (truth == 0)).sum())
        accuracy = (TP + TN) / n_val if n_val > 0 else 0.0
        positive_rate = pred.mean()
        f1 = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0.0
        metrics[concept] = dict(
            accuracy=accuracy,
            positive_rate=positive_rate,
            F1=f1,
            TP=TP,
            FP=FP,
            FN=FN,
            TN=TN
        )

    # Filtrage
    if f1_cutoff is not None:
        filtered = [c for c, m in metrics.items() if m['F1'] >= f1_cutoff]
    else:
        filtered = list(metrics.keys())

    # Sauvegarde des métriques
    if save_dir and config is not None:
        out_file = os.path.join(
            save_dir,
            f"detection_concept_{config.cavs_type}_{config.annotation}_{config.agg_mode}_{config.agg_scope}.json"
        )
        with open(out_file, 'w') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=4)

    return train_df, val_df, cosine_df, thresholds, metrics, filtered




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
    
    groundtruth_df.rename(columns=lambda x: clean_concept_name(x), inplace=True)
    groundtruth_df.columns = ['concept_' + col if col not in ['text', 'label'] else col for col in groundtruth_df.columns]
    # print("groundtruth_df.columns", groundtruth_df.columns)
    
    # Effectuer une jointure sur les colonnes 'text' et 'label'
    merged_df = cosine_df.merge(groundtruth_df, on=['text', 'label'], how='inner')
    print("merged_df.columns", merged_df.columns)

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
