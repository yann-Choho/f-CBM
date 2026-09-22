"""
Script spécialisé pour la nouvelle heuristique en mode image-only
Basé sur new_heuristique_multimodal.py adapté pour les données visuelles uniquement
Combine détection de concepts, scoring et ranking pour les images
"""

import pickle
import os
import gc
import time
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score, accuracy_score, r2_score, mean_squared_error, mean_absolute_error


# =========================================================================
# Fonction utilitaire : Similarité cosinus cubée (issue de CB_LLM)
# =========================================================================

def cos_sim_cubed(cbl_features, target):
    """
    Calcule la similarité cosinus cubée entre deux tenseurs.
    Utilisée pour améliorer la séparabilité des concepts.
    """
    cbl_features = cbl_features - torch.mean(cbl_features, dim=-1, keepdim=True)
    target = target - torch.mean(target, dim=-1, keepdim=True)

    cbl_features = F.normalize(cbl_features**3, dim=-1)
    target = F.normalize(target**3, dim=-1)

    sim = torch.sum(cbl_features * target, dim=-1)
    return sim.mean()


def clean_concept_name(name):
    """
    Nettoie un nom de concept en supprimant les préfixes et espaces superflus
    """
    import re
    name = name.replace("cos_", "").replace("concept_", "").replace("dummy_", "").strip()
    name = re.sub(r"Let me know.*", "", name)  # Supprime les phrases inutiles
    return " ".join(name.split())  # Réduit les espaces multiples


# =========================================================================
# Fonction principale : Calcul des similarités cosinus et métriques
# =========================================================================

def compute_cosine_matrix_and_metrics_image_dataloader(
    dataloader,
    model,
    cavs,
    f1_cutoff=None,
    device=torch.device("cuda"),
    save_dir=None,
    config=None,
    annotation=None,
    cos_cubed=False
):
    """
    Calcule les similarités cosinus entre embeddings d'images et vecteurs CAV,
    puis génère un split train/val 70/30 stratifié, définit des seuils via la médiane,
    et calcule les métriques (accuracy, positive_rate, F1, TP, FP, FN, TN) sur validation.
    
    Version image-only : pas de texte, uniquement images
    
    Args:
        dataloader: DataLoader fournissant 'image', 'label', et 'concept_{c}' pour chaque concept
        model: Modèle vision avec méthode get_pooled_output ou get_image_features
        cavs (dict): Vecteurs CAV par concept
        f1_cutoff (float, optional): Seuil minimal de F1 pour filtrer les concepts
        device: Device pour le calcul
        save_dir: Répertoire de sauvegarde
        config: Configuration pour noms de fichiers
        annotation: Suffixe pour nom de fichier pickle
        cos_cubed (bool): Utiliser la similarité cosinus cubée
    
    Returns:
        train_df, val_df, cosine_df, thresholds, metrics, filtered_concepts
    """
    # Préparation des chemins de sauvegarde
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        cosine_path = os.path.join(save_dir, f"cosine_df_{annotation}.pkl")
    else:
        cosine_path = None
    
    # Chargement ou calcul du DataFrame de similarités
    if cosine_path and os.path.exists(cosine_path):
        print(f"📂 Chargement des similarités depuis: {cosine_path}")
        with open(cosine_path, 'rb') as f:
            cosine_df = pickle.load(f)
    else:
        print("🖼️ Calcul des similarités cosinus pour les images...")
        model.to(device).eval()
        records = []
        
        for batch in tqdm(dataloader, desc="Cosinus par batch", unit="batch"):
            images = batch['image'].to(device)
            
            # Extraction des embeddings d'images
            with torch.no_grad():
                if hasattr(model, 'get_pooled_output'):
                    # Mode image-only avec get_pooled_output
                    cls_emb = model.get_pooled_output(pixel_values=images)
                elif hasattr(model, 'get_image_features'):
                    # CLIP image features
                    cls_emb = model.get_image_features(pixel_values=images)
                elif hasattr(model, 'vision_model'):
                    # BLIP ou autres avec vision_model
                    cls_emb = model.vision_model(pixel_values=images).pooler_output
                else:
                    raise ValueError("Méthode d'extraction d'embeddings d'images non reconnue")
            
            # Calcul des similarités par image
            for idx, emb in enumerate(cls_emb):
                sample = {'image_idx': idx}
                
                # Vérité terrain et similarité par concept
                for concept, cav_vec in cavs.items():
                    # Vérité terrain pour ce concept
                    concept_key = _find_concept_key_in_batch(concept, batch)
                    if concept_key:
                        gt = int(batch[concept_key][idx].item())
                        sample[f'concept_{concept}'] = gt
                    else:
                        sample[f'concept_{concept}'] = 0
                    
                    # Calcul de la similarité
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
            print(f"💾 Similarités sauvegardées: {cosine_path}")
    
    # Colonnes de similarités et vérité terrain
    sim_cols = [c for c in cosine_df.columns if c.startswith('sim_')]
    gt_cols = [c for c in cosine_df.columns if c.startswith('concept_')]
    
    print(f"📊 {len(sim_cols)} concepts détectés")
    
    # Split 70/30 stratifié
    strat = (cosine_df[gt_cols].sum(axis=1) > 0).astype(int)
    train_df, val_df = train_test_split(
        cosine_df,
        test_size=0.3,
        random_state=42,
        stratify=strat
    )
    
    print(f"✂️ Split: {len(train_df)} train / {len(val_df)} val")
    
    # Seuils par concept = médiane sur train
    thresholds = {c: train_df[c].median() for c in sim_cols}
    
    # Prédictions sur validation
    preds = val_df[sim_cols].gt(pd.Series(thresholds)).astype(int)
    
    # Calcul des métriques par concept
    metrics = {}
    n_val = len(val_df)
    
    for sim_c in sim_cols:
        concept = sim_c.replace('sim_', '')
        concept_clean = clean_concept_name(concept)  # Nettoie 'cos_', 'concept_', etc.
        truth = val_df[f'concept_{concept_clean}']
        pred = preds[sim_c]
        
        TP = int(((pred == 1) & (truth == 1)).sum())
        FP = int(((pred == 1) & (truth == 0)).sum())
        FN = int(((pred == 0) & (truth == 1)).sum())
        TN = int(((pred == 0) & (truth == 0)).sum())
        
        accuracy = (TP + TN) / n_val if n_val > 0 else 0.0
        positive_rate = pred.mean()
        f1 = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0.0
        
        metrics[concept_clean] = dict(
            accuracy=accuracy,
            positive_rate=positive_rate,
            F1=f1,
            TP=TP,
            FP=FP,
            FN=FN,
            TN=TN,
            threshold=thresholds[sim_c]
        )
    
    # Filtrage par F1
    if f1_cutoff is not None:
        filtered = [c for c, m in metrics.items() if m['F1'] >= f1_cutoff]
        print(f"🔍 {len(filtered)}/{len(metrics)} concepts avec F1 >= {f1_cutoff}")
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
        print(f"💾 Métriques sauvegardées: {out_file}")
    
    return train_df, val_df, cosine_df, thresholds, metrics, filtered


def _find_concept_key_in_batch(concept_name, batch):
    """Trouve la clé correspondant au concept dans le batch"""
    possible_keys = [
        concept_name,
        f"concept_{concept_name}",
    ]
    
    for key in possible_keys:
        if key in batch:
            return key
    return None


# =========================================================================
# Fonction : Calcul des fréquences de concepts
# =========================================================================

def compute_concept_frequencies_image(dataloader, concept_list=None):
    """
    Calcule la fréquence de présence des concepts dans les images
    
    Args:
        dataloader: DataLoader des images
        concept_list: Liste des concepts (détection auto si None)
    
    Returns:
        dict: Fréquences par concept
    """
    print("📊 Calcul des fréquences des concepts...")
    
    if concept_list is None:
        # Détection automatique
        sample_batch = next(iter(dataloader))
        concept_list = [k for k in sample_batch.keys() if k.startswith(("concept_"))]
        print(f"📝 {len(concept_list)} concepts détectés")
    
    from collections import defaultdict
    concept_counts = defaultdict(int)
    total_samples = 0
    
    for batch in tqdm(dataloader, desc="Comptage", unit="batch"):
        batch_size = batch["image"].size(0)
        total_samples += batch_size
        
        for concept_key in concept_list:
            if concept_key in batch:
                concept_counts[concept_key] += batch[concept_key].sum().item()
    
    # Fréquences relatives
    frequencies = {}
    for concept_key, count in concept_counts.items():
        clean_name = clean_concept_name(concept_key)
        frequencies[clean_name] = count / total_samples if total_samples > 0 else 0.0
    
    # Top fréquences
    sorted_freq = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)
    print("\n🏆 Top 10 concepts les plus fréquents:")
    for i, (concept, freq) in enumerate(sorted_freq[:10]):
        print(f"  {i+1:2d}. {concept:20} | {freq:.3f}")
    
    return frequencies


# =========================================================================
# Visualisation : Scatter plot concept vs threshold
# =========================================================================

def plot_concept_threshold_image(cosine_df, groundtruth_df, concept, thresholds, gt_prefix="concept_"):
    """
    Affiche un scatter plot pour un concept donné en mode image.
    Les points sont colorés selon la ground truth (vert=1, rouge=0).
    
    Args:
        cosine_df: DataFrame avec similarités
        groundtruth_df: DataFrame avec vérité terrain
        concept: Nom du concept
        thresholds: Dictionnaire des seuils
        gt_prefix: Préfixe des colonnes ground truth
    """
    # Nettoyage des noms
    groundtruth_df.rename(columns=lambda x: clean_concept_name(x), inplace=True)
    groundtruth_df.columns = [
        'concept_' + col if col not in ['image_idx', 'label'] else col 
        for col in groundtruth_df.columns
    ]
    
    # Jointure sur index d'image
    merged_df = cosine_df.merge(groundtruth_df, left_index=True, right_index=True, how='inner')
    
    gt_col = f"{gt_prefix}{concept}"
    if gt_col not in merged_df.columns:
        if concept in merged_df.columns:
            gt_col = concept
        else:
            raise ValueError(f"Colonne ground truth pour '{concept}' non trouvée")
    
    if concept not in merged_df.columns:
        raise ValueError(f"Colonne de similarité pour '{concept}' non trouvée")
    
    sims = merged_df[concept]
    truth = merged_df[gt_col]
    colors = truth.map({1: "green", 0: "red"})
    
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(sims)), sims, c=colors, alpha=0.7, edgecolor='k', s=10)
    plt.axhline(y=thresholds[concept], color="blue", linestyle="--", 
                label=f"Threshold = {thresholds[concept]:.3f}")
    plt.xlabel("Échantillon (Image)")
    plt.ylabel("Cosine Similarity")
    plt.title(f"Cosine Similarity - Concept '{concept}' (Mode Image)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()
