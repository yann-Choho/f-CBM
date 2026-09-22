"""
Script spécialisé pour le ranking LIG (Layer Integrated Gradients) en mode image-only
Calcul des gradients intégrés pour les couches vision des modèles multimodaux
Basé sur l'implémentation LIG_ranking_multimodal.py adaptée pour images uniquement
"""

import os
import json
import torch
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc
import torch.nn.functional as F
from torch.cuda.amp import autocast
from captum.attr import LayerIntegratedGradients
from collections import defaultdict

def compute_attributions_from_dataloader(dataloader, model, lig, device):
    """
    Calcule les attributions LIG sur les images uniquement
    
    Args:
        dataloader: DataLoader contenant les images
        model: Modèle vision avec get_pooled_output ou get_image_features
        lig: Instance de LayerIntegratedGradients configurée
        device: Device (cuda ou cpu)
    
    Returns:
        pd.DataFrame: DataFrame contenant les attributions calculées
    """
    attributions = []
    print("🖼️ Début du calcul des attributions LIG pour les images...")

    model_dtype = next(model.parameters()).dtype
    
    for batch in tqdm(dataloader, desc="Attributions LIG", unit="batch"):
        # Extraction des images et labels
        images = batch["image"].to(device).to(model_dtype)
        labels = batch["label"].to(device)
        
        # Calcul du pooled_output avec gradient activé
        pooled_output = _get_image_pooled_output(model, images, device)
        pooled_output = pooled_output.to(torch.float32)
        pooled_output.requires_grad_(True)
        
        # Baseline: vecteur zéro de même forme
        baseline = torch.zeros_like(pooled_output)
        
        # Calcul des attributions avec LIG
        ctx = torch.autocast(device_type="cuda", dtype=torch.float32) if device.type == "cuda" else torch.no_grad()
        with torch.enable_grad(), ctx:
            attr, delta = lig.attribute(
                inputs=pooled_output,
                baselines=baseline,
                target=labels,
                n_steps=50,
                internal_batch_size=1,
                return_convergence_delta=True
            )
        
        # Sauvegarde des attributions
        for a in attr:
            attributions.append(a.detach().cpu())
        
        # Libération mémoire
        del pooled_output, baseline, attr
        torch.cuda.empty_cache()
        gc.collect()
    
    print("✅ Calcul des attributions terminé!")
    return pd.DataFrame({"attributions": attributions})


def _get_image_pooled_output(model, images, device):
    """
    Extrait le pooled output des images selon le type de modèle
    
    Args:
        model: Modèle vision
        images: Batch d'images
        device: Device
    
    Returns:
        torch.Tensor: Pooled output des images
    """
    if hasattr(model, 'get_pooled_output'):
        # Modèle avec méthode get_pooled_output (mode image-only)
        return model.get_pooled_output(pixel_values=images)
    elif hasattr(model, 'get_image_features'):
        # CLIP image features
        return model.get_image_features(pixel_values=images)
    elif hasattr(model, 'vision_model'):
        # BLIP ou autres avec vision_model
        return model.vision_model(pixel_values=images).pooler_output
    else:
        raise ValueError("Impossible d'extraire les features d'images du modèle")


def compute_similarity_cav_image(attribution, cav, device):
    """
    Calcule la similarité cosinus entre les attributions et un vecteur CAV
    pour les images
    
    Args:
        attribution: Tenseur d'attributions (batch, dim) sur GPU
        cav: Vecteur CAV sur le device
        device: Device
    
    Returns:
        torch.Tensor: Similarité cosinus (1D, taille = batch)
    """
    cav = cav.unsqueeze(0).to(device)
    att = attribution.to(device)
    
    # Calcul de la similarité cosinus
    similarity = F.cosine_similarity(cav, att, dim=1)
    return similarity


def compute_cosine_similarities(attributions_df, df, cavs, device):
    """
    Calcule les similarités cosinus entre attributions et CAVs pour chaque image
    
    Args:
        attributions_df: DataFrame contenant les attributions
        df: DataFrame original à mettre à jour
        cavs: Dictionnaire des vecteurs CAV
        device: Device
    
    Returns:
        pd.DataFrame: DataFrame mis à jour avec colonnes 'cos_{concept}'
    """
    # Initialisation des colonnes
    for concept in cavs.keys():
        df[f'cos_{concept}'] = 0.0
    
    print("🔄 Calcul des similarités cosinus avec les CAVs...")
    for i in tqdm(range(attributions_df.shape[0]), desc='Cosinus', unit='row'):
        attrib = attributions_df.at[i, 'attributions'].to(device)
        
        # Calcul pour chaque concept
        for concept, cav in cavs.items():
            sim = compute_similarity_cav_image(attrib.unsqueeze(0), cav, device)
            df.at[i, f'cos_{concept}'] = sim.item()
        
        # Libération mémoire
        del attrib
        torch.cuda.empty_cache()
        gc.collect()
    
    print("✅ Calcul des similarités terminé!")
    return df


def postprocess_cosine(df, cavs_keys, mode="abs", agg_scope="all"):
    """
    Post-traitement des similarités cosinus pour les images
    
    Args:
        df: DataFrame avec colonnes 'cos_{concept}'
        cavs_keys: Liste des concepts
        mode: "abs" ou "clip" pour traiter les valeurs négatives
        agg_scope: "all" ou "present" pour l'agrégation
    
    Returns:
        tuple: (df mis à jour, sorted_concepts)
    """
    cosine_columns = [f'cos_{concept}' for concept in cavs_keys]
    
    # Traitement des valeurs négatives
    if mode == "abs":
        df[cosine_columns] = df[cosine_columns].abs()
    elif mode == "clip":
        df[cosine_columns] = df[cosine_columns].clip(lower=0)
    else:
        raise ValueError(f"Mode inconnu: {mode}. Utilisez 'abs' ou 'clip'.")
    
    # Agrégation par concept
    aggregated_scores = {}
    
    for concept in cavs_keys:
        col_name = f'cos_{concept}'
        
        if agg_scope == "all":
            # Moyenne sur toutes les lignes
            mean_score = df[col_name].mean()
        elif agg_scope == "present":
            # Moyenne uniquement sur les lignes où le concept est présent
            # Recherche de la colonne du concept
            concept_col = _find_concept_column(df, concept)
            if concept_col and concept_col in df.columns:
                mask = df[concept_col] == 1
                if mask.sum() > 0:
                    mean_score = df.loc[mask, col_name].mean()
                else:
                    mean_score = 0.0
            else:
                # Si pas de colonne concept, utiliser la moyenne globale
                mean_score = df[col_name].mean()
        else:
            raise ValueError(f"agg_scope inconnu: {agg_scope}")
        
        aggregated_scores[col_name] = mean_score
    

    # # Fonction pour nettoyer les noms des concepts
    def clean_concept_name(name):
        import re
        name = name.replace("cos_", "").replace("dummy_ ", "").strip()
        name = re.sub(r"Let me know.*", "", name)  # Supprime les phrases inutiles
        return " ".join(name.split())  # Réduit les espaces multiples

    # # Appliquer le nettoyage sur la liste chargée
    # sorted_concepts = [(clean_concept_name(name), score) for name, score in aggregated_scores] # expérimental

    # # # Tri des concepts par score décroissant
    # sorted_concepts = sorted(aggregated_scores.items(), key=lambda x: x[1], reverse=True)

    # Tri ET nettoyage en une seule opération
    sorted_concepts = sorted(
        [(clean_concept_name(name), score) for name, score in aggregated_scores.items()],
        key=lambda x: x[1],
        reverse=True
    )

    return df, sorted_concepts


def _find_concept_column(df, concept_name):
    """Trouve la colonne correspondant au concept dans le DataFrame"""
    possible_names = [
        concept_name,
        f"concept_{concept_name}",
        f"dummy_{concept_name}"
    ]
    
    for name in possible_names:
        if name in df.columns:
            return name
    return None


def create_lig_instance_for_image_model(model, device):
    """
    Crée une instance de LayerIntegratedGradients adaptée au modèle vision
    
    Args:
        model: Modèle vision avec classifier
        device: Device
    
    Returns:
        LayerIntegratedGradients: Instance LIG configurée
    """
    # Détection de la couche cible
    if hasattr(model, 'concat_layer'):
        # Modèle avec concat_layer (cas multimodal adapté)
        target_layer = model.concat_layer
        
        def forward_func(pooled):
            pooled = model.concat_layer(pooled)
            logits = model.classifier(pooled)
            return logits
            
    elif hasattr(model, 'visual_projection'):
        # CLIP avec projection visuelle
        target_layer = model.visual_projection
        
        def forward_func(features):
            projected = model.visual_projection(features)
            logits = model.classifier(projected)
            return logits
            
    elif hasattr(model, 'classifier'):
        # Modèle avec classifier direct
        target_layer = model.classifier
        
        def forward_func(features):
            return model.classifier(features)
    else:
        raise ValueError("Impossible de détecter une couche appropriée pour LIG")
    
    lig = LayerIntegratedGradients(forward_func, layer=target_layer)
    
    print(f"✅ LIG configuré avec la couche: {type(target_layer).__name__}")
    return lig


def main_lig_ranking_image(image_loader, model, config, cavs_dict, 
                           mode="abs", agg_scope="all", batch_size=4):
    """
    Pipeline complet LIG pour le ranking des concepts en mode image
    
    Args:
        image_loader: DataLoader des images
        model: Modèle vision avec classifier
        config: Configuration du projet
        cavs_dict: Dictionnaire des CAVs
        mode: Mode de traitement ("abs" ou "clip")
        agg_scope: Scope d'agrégation ("all" ou "present")
        batch_size: Taille des batches
    
    Returns:
        dict: Ranking des concepts par score LIG
    """
    import time
    start_time = time.time()
    
    print("🚀 Lancement du pipeline LIG complet pour les images")
    print("=" * 60)
    
    device = config.device
    model.to(device)
    model.eval()
    
    # 1. Calcul ou chargement des attributions
    path_attr = os.path.join(
        config.SAVE_PATH, "blue_checkpoints", config.model_name, 
        "cavs", config.cavs_type, 
        f"attributions_df_image_{config.cavs_type}_{config.annotation}.pkl"
    )
    os.makedirs(os.path.dirname(path_attr), exist_ok=True)
    
    if os.path.exists(path_attr):
        print(f"📂 Chargement des attributions depuis: {path_attr}")
        with open(path_attr, "rb") as f:
            attributions_df = pickle.load(f)
    else:
        print("🔄 Calcul des attributions...")
        lig = create_lig_instance_for_image_model(model, device)
        attributions_df = compute_attributions_from_image_dataloader(
            image_loader, model, lig, device
        )
        with open(path_attr, "wb") as f:
            pickle.dump(attributions_df, f)
        print(f"💾 Attributions sauvegardées: {path_attr}")
    
    # 2. Conversion des CAVs en tenseurs GPU
    print("🔄 Conversion des CAVs...")
    cavs_tensors = {}
    for k, v in cavs_dict.items():
        cavs_tensors[k] = torch.tensor(v, dtype=torch.float32).to(device)
    print(f"✅ {len(cavs_tensors)} CAVs convertis")
    
    # 3. Création d'un DataFrame temporaire pour les calculs
    # (on suppose que image_loader.dataset a les infos nécessaires)
    temp_df = pd.DataFrame({'index': range(len(attributions_df))})
    
    # Tentative d'extraction des labels de concepts depuis le dataset
    try:
        dataset = image_loader.dataset
        for concept in cavs_dict.keys():
            concept_col = _find_concept_column_in_dataset(dataset, concept)
            if concept_col:
                temp_df[concept_col] = [
                    dataset[i].get(concept_col, 0) for i in range(len(dataset))
                ]
    except Exception as e:
        print(f"⚠️ Impossible d'extraire les labels de concepts: {e}")
    
    # 4. Calcul des similarités cosinus
    print("🔄 Calcul des similarités cosinus...")
    temp_df = compute_cosine_similarities(
        attributions_df, temp_df, cavs_tensors, device
    )
    
    # 5. Post-traitement et ranking
    print("🔄 Post-traitement et ranking...")
    temp_df, sorted_concepts = postprocess_cosine(
        temp_df, list(cavs_dict.keys()), mode=mode, agg_scope=agg_scope
    )
    
    # 6. Sauvegarde des résultats
    save_dir = os.path.join(
        config.SAVE_PATH, "blue_checkpoints", config.model_name,
        "cavs", config.cavs_type
    )
    os.makedirs(save_dir, exist_ok=True)
    
    annotation_suffix = f"_{config.annotation}" if hasattr(config, 'annotation') and config.annotation else ""
    ranking_file = os.path.join(
        save_dir, 
        f"sorted_macro_concepts_lig_image_{mode}_{agg_scope}{annotation_suffix}.json"
    )
    
    with open(ranking_file, "w") as f:
        json.dump(sorted_concepts, f, indent=2)
    print(f"💾 Ranking sauvegardé: {ranking_file}")
    
    # 7. Affichage des résultats
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"🎉 Pipeline LIG terminé en {elapsed_time:.2f} secondes")
    print(f"\n🏆 Top 10 concepts par score LIG:")
    for i, (concept, score) in enumerate(sorted_concepts[:10]):
        clean_name = concept.replace('cos_', '')
        print(f"  {i+1:2d}. {clean_name:20} | Score: {score:.4f}")
    
    # Conversion en dictionnaire pour retour
    ranking_dict = {concept.replace('cos_', ''): score for concept, score in sorted_concepts}
    
    # Libération mémoire
    del attributions_df, cavs_tensors, temp_df
    gc.collect()
    torch.cuda.empty_cache()
    
    return ranking_dict


def _find_concept_column_in_dataset(dataset, concept_name):
    """Trouve la colonne du concept dans le dataset"""
    try:
        sample = dataset[0]
        possible_names = [
            concept_name,
            f"concept_{concept_name}",
            f"dummy_{concept_name}"
        ]
        for name in possible_names:
            if name in sample:
                return name
    except:
        pass
    return None


def load_lig_ranking_image(config, mode="abs", agg_scope="all"):
    """
    Charge un ranking LIG précédemment sauvegardé pour les images
    
    Args:
        config: Configuration du projet
        mode: Mode utilisé ("abs" ou "clip")
        agg_scope: Scope utilisé ("all" ou "present")
    
    Returns:
        dict: Ranking chargé ou None
    """
    save_dir = os.path.join(
        config.SAVE_PATH, "blue_checkpoints", config.model_name,
        "cavs", config.cavs_type
    )
    
    annotation_suffix = f"_{config.annotation}" if hasattr(config, 'annotation') and config.annotation else ""
    ranking_file = os.path.join(
        save_dir, 
        f"sorted_macro_concepts_lig_image_{mode}_{agg_scope}{annotation_suffix}.json"
    )
    
    if os.path.exists(ranking_file):
        with open(ranking_file, 'r') as f:
            ranking_list = json.load(f)
        
        # Conversion en dictionnaire
        ranking_dict = {concept.replace('cos_', ''): score for concept, score in ranking_list}
        print(f"📂 Ranking LIG chargé: {ranking_file}")
        return ranking_dict
    else:
        print(f"❌ Fichier ranking non trouvé: {ranking_file}")
        return None


if __name__ == "__main__":
    print("Script LIG_ranking_image.py - Mode image uniquement")
    print("=" * 60)
    print("\nExemple d'utilisation complète :")
    print("""
    # 1. Chargement des données et modèle
    from unified_config import Config
    config = Config(model_name='clip', dataset='n24news')
    
    # 2. Chargement des CAVs
    from mean_cavs_creation_image import load_cavs_image_only
    cavs = load_cavs_image_only(config)
    
    # 3. Calcul du ranking LIG
    ranking = main_lig_ranking_image(
        image_loader=train_loader,
        model=vision_model,
        config=config,
        cavs_dict=cavs,
        mode="abs",
        agg_scope="all"
    )
    
    # 4. Affichage des résultats
    for concept, score in list(ranking.items())[:10]:
        print(f"{concept}: {score:.4f}")
    """)