"""
Module unifié pour le calcul des CAVs (Concept Activation Vectors) 
et du score R² d'identifiabilité.

Supporte 4 modalités :
- Text (DataFrame-based)
- Image (DataLoader-based)
- Multimodal (DataLoader-based)
- Gemma (DataFrame-based)

Méthodes :
- mean_minus_others : CAV = moyenne(positifs) - moyenne(négatifs)
- regression : CAV = coefficients de régression linéaire sur cosine similarity
"""

import os
import json
import pickle
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# ============================================================================
# 1. EXTRACTION DES EMBEDDINGS (Helpers communs)
# ============================================================================

def _extract_embeddings_from_dataframe(
    df, 
    baseline_model, 
    tokenizer, 
    config,
    text_column='text',
    is_gemma=False
):
    """
    Extrait les embeddings pour chaque texte d'un DataFrame.
    
    Args:
        df: DataFrame contenant au moins une colonne 'text'
        baseline_model: Modèle avec méthode get_pooled_output
        tokenizer: Tokenizer associé
        config: Config avec device, max_len
        text_column: Nom de la colonne contenant le texte
        is_gemma: Si True, utilise get_pooled_output(input_ids) sans attention_mask
        
    Returns:
        pd.DataFrame: DataFrame avec colonne 'embeddings' ajoutée
    """
    device = config.device
    baseline_model.to(device).eval()
    
    df_copy = df.copy()
    df_copy['embeddings'] = None
    df_copy['embeddings'] = df_copy['embeddings'].astype(object)
    
    print(f"🔄 Extraction des embeddings (DataFrame, {len(df)} samples)...")
    
    for idx, row in tqdm(df_copy.iterrows(), total=len(df_copy), desc="Embeddings"):
        text = row[text_column]
        
        # Tokenization
        inputs = tokenizer(
            text,
            max_length=config.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        # Forward pass
        with torch.no_grad():
            if is_gemma:
                outputs = baseline_model.get_pooled_output(input_ids)
            else:
                outputs = baseline_model.get_pooled_output(input_ids, attention_mask)
        
        # Stockage CPU
        embedding_cpu = outputs.flatten().detach().cpu()
        df_copy.at[idx, 'embeddings'] = embedding_cpu
        
        # Cleanup
        del input_ids, attention_mask, outputs
        torch.cuda.empty_cache()
    
    return df_copy


def _extract_embeddings_from_loader_image(
    loader,
    baseline_model,
    config,
    concept_list=None
):
    """
    Extrait les embeddings d'images depuis un DataLoader.
    
    Args:
        loader: DataLoader retournant dict avec 'image' et 'concept_*'
        baseline_model: Modèle vision (CLIP/BLIP)
        config: Config avec device
        concept_list: Liste des concepts (détectée auto si None)
        
    Returns:
        tuple: (pos_embeds, neg_embeds, concept_list)
            - pos_embeds: dict {concept: [embeddings positifs]}
            - neg_embeds: dict {concept: [embeddings négatifs]}
            - concept_list: liste des concepts traités
    """
    device = config.device
    baseline_model.to(device).eval()
    
    # Détection automatique des concepts
    if concept_list is None:
        first_batch = next(iter(loader))
        concept_list = [k for k in first_batch.keys() if k.startswith("concept_") ]
        
        # Reconstruction du loader après lecture du premier batch
        loader = torch.utils.data.DataLoader(
            loader.dataset, 
            batch_size=loader.batch_size,
            shuffle=False, 
            num_workers=getattr(loader, 'num_workers', 0),
            pin_memory=True
        )
        print(f"📝 Concepts détectés : {len(concept_list)}")
    
    pos_embeds = defaultdict(list)
    neg_embeds = defaultdict(list)
    
    print(f"🔄 Extraction des embeddings (Image DataLoader, {len(loader)} batches)...")
    
    for batch in tqdm(loader, desc="Image embeddings"):
        images = batch["image"].to(device)
        batch_size = images.size(0)

        # Forward pass vision
        with torch.no_grad():
            if hasattr(baseline_model, 'get_pooled_output'):
                pooled_output = baseline_model.get_pooled_output(pixel_values=images)
            elif hasattr(baseline_model, 'get_image_features'):
                pooled_output = baseline_model.get_image_features(pixel_values=images)
            elif hasattr(baseline_model, 'vision_model'):
                pooled_output = baseline_model.vision_model(pixel_values=images).pooler_output
            else:
                raise ValueError("Méthode d'extraction embeddings image non reconnue")
        
        pooled_output_cpu = pooled_output.cpu()
        
        # Dispatch pos/neg
        for i in range(batch_size):
            for cname in concept_list:
                # Normalisation du nom du concept
                if not cname.startswith("concept_"):
                    cname = "concept_" + cname

                if batch[cname][i] == 1:
                    pos_embeds[cname].append(pooled_output_cpu[i])
                else:
                    neg_embeds[cname].append(pooled_output_cpu[i])
                    
        torch.cuda.empty_cache()
    
    return pos_embeds, neg_embeds, concept_list


def _extract_embeddings_from_loader_multimodal(
    loader,
    baseline_model,
    processor,
    config,
    concept_list=None,
    use_images=True
):
    """
    Extrait les embeddings multimodaux (texte + image) depuis un DataLoader.
    
    Args:
        loader: DataLoader retournant dict avec 'text', 'image', 'concept_*'
        baseline_model: Modèle multimodal (CLIP/BLIP)
        processor: Processor pour prétraiter text + images
        config: Config avec device
        concept_list: Liste des concepts (détectée auto si None)
        use_images: Si True, utilise text+image; sinon text seul
        
    Returns:
        tuple: (pos_embeds, neg_embeds, concept_list)
    """
    device = config.device
    baseline_model.to(device).eval()
    
    # Détection automatique des concepts
    if concept_list is None:
        first_batch = next(iter(loader))
        concept_list = [k for k in first_batch.keys() if k.startswith("concept_")]
        
        # Reconstruction du loader
        loader = torch.utils.data.DataLoader(
            loader.dataset,
            batch_size=loader.batch_size,
            shuffle=False,
            num_workers=getattr(loader, 'num_workers', 0),
            pin_memory=True
        )
    
    pos_embeds = defaultdict(list)
    neg_embeds = defaultdict(list)
    
    print(f"🔄 Extraction des embeddings (Multimodal DataLoader, {len(loader)} batches)...")
    
    for batch in tqdm(loader, desc="Multimodal embeddings"):
        texts = list(batch["text"])
        
        # Préparation inputs
        if use_images:
            imgs = batch["image"]
            batch_inputs = processor(
                text=texts,
                images=imgs,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(device)
        else:
            batch_inputs = processor(
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(device)
        
        # Forward pass
        with torch.no_grad():
            pooled_output = baseline_model.get_pooled_output(
                batch_inputs.input_ids,
                batch_inputs.attention_mask,
                batch_inputs.pixel_values if use_images else None
            )
        
        pooled_output_cpu = pooled_output.cpu()
        
        # Dispatch pos/neg
        for i in range(len(texts)):
            for cname in concept_list:
                # Normalisation du nom du concept
                if not (cname.startswith("concept_")):
                    cname = "concept_" + cname
                
                if batch[cname][i] == 1:
                    pos_embeds[cname].append(pooled_output_cpu[i])
                else:
                    neg_embeds[cname].append(pooled_output_cpu[i])
        
        torch.cuda.empty_cache()
    
    return pos_embeds, neg_embeds, concept_list


# ============================================================================
# 2. CALCUL DES CAVs PAR MEAN-MINUS-OTHERS
# ============================================================================

def compute_cavs_mean_minus_text(
    df_aug,
    baseline_model,
    tokenizer,
    config,
    text_column='text'
):
    """
    Calcule les CAVs pour modalité TEXT via méthode mean-minus-others.
    
    Args:
        df_aug: DataFrame avec colonnes 'text', 'label', 'concept_*'
        baseline_model: Modèle avec get_pooled_output(input_ids, attention_mask)
        tokenizer: Tokenizer associé
        config: Config avec device, max_len, SAVE_PATH, etc.
        text_column: Nom de la colonne texte
        
    Returns:
        dict: {concept_name: cav_vector (numpy array)}
    """
    print("=" * 60)
    print("🔵 CALCUL CAVs - MODALITÉ TEXT (mean-minus-others)")
    print("=" * 60)
    
    device = config.device
    baseline_model.to(device)
    
    # Consolidation des concepts
    concept_name_list = []
    df_aug['consolidated_concepts'] = [[] for _ in range(len(df_aug))]
    
    for column in df_aug.columns:
        if 'concept' not in column:
            continue
        concept_name = column.replace('concept_', '')
        concept_name_list.append(concept_name)
        for i in df_aug.loc[df_aug[column] == 1].index:
            df_aug.at[i, 'consolidated_concepts'].append(concept_name)
    
    print(f"📝 {len(concept_name_list)} concepts détectés : {concept_name_list}")
    
    # Extraction des embeddings
    df_with_embeddings = _extract_embeddings_from_dataframe(
        df_aug, baseline_model, tokenizer, config, text_column, is_gemma=False
    )
    
    # Calcul des CAVs (CPU uniquement)
    cavs_concepts = {}
    
    print("🧮 Calcul des CAVs...")
    for concept in tqdm(concept_name_list, desc="CAVs"):
        # Sélection positifs
        concept_data = df_with_embeddings.loc[
            df_with_embeddings['consolidated_concepts'].apply(lambda x: concept in x)
        ]
        
        if concept_data.empty:
            print(f"⚠️  Pas d'échantillons pour '{concept}'")
            continue
        
        concept_embeddings = torch.stack([emb for emb in concept_data['embeddings'].values]).cpu()
        
        # Sélection négatifs
        other_data = df_with_embeddings.loc[
            df_with_embeddings['consolidated_concepts'].apply(lambda x: concept not in x)
        ]
        
        if other_data.empty:
            print(f"⚠️  Pas d'échantillons négatifs pour '{concept}'")
            continue
        
        other_embeddings = torch.stack([emb for emb in other_data['embeddings'].values]).cpu()
        
        # CAV = moyenne(pos) - moyenne(neg)
        cav = concept_embeddings.mean(dim=0) - other_embeddings.mean(dim=0)
        cavs_concepts[concept] = cav.numpy()
        
        print(f"✅ CAV '{concept}' : pos={len(concept_data)}, neg={len(other_data)}")
    
    # Sauvegarde
    _save_cavs_json(cavs_concepts, config, mode='text')
    
    return cavs_concepts


def compute_cavs_mean_minus_gemma(
    df_aug,
    baseline_model,
    tokenizer,
    config,
    text_column='text'
):
    """
    Calcule les CAVs pour modalité GEMMA via méthode mean-minus-others.
    Identique à TEXT sauf que get_pooled_output ne prend que input_ids.
    
    Args:
        df_aug: DataFrame avec colonnes 'text', 'label', 'dummy_*'
        baseline_model: Modèle Gemma avec get_pooled_output(input_ids)
        tokenizer: Tokenizer associé
        config: Config avec device, max_len, SAVE_PATH, etc.
        text_column: Nom de la colonne texte
        
    Returns:
        dict: {concept_name: cav_vector (numpy array)}
    """
    print("=" * 60)
    print("🟣 CALCUL CAVs - MODALITÉ GEMMA (mean-minus-others)")
    print("=" * 60)
    
    device = config.device
    baseline_model.to(device)
    
    # Consolidation des concepts
    concept_name_list = []
    df_aug['consolidated_concepts'] = [[] for _ in range(len(df_aug))]
    
    for column in df_aug.columns:
        if 'concept' not in column:
            continue
        concept_name = column.replace('concept_', '')
        concept_name_list.append(concept_name)
        for i in df_aug.loc[df_aug[column] == 1].index:
            df_aug.at[i, 'consolidated_concepts'].append(concept_name)
    
    print(f"📝 {len(concept_name_list)} concepts détectés : {concept_name_list}")
    
    # Extraction des embeddings (spécifique Gemma)
    df_with_embeddings = _extract_embeddings_from_dataframe(
        df_aug, baseline_model, tokenizer, config, text_column, is_gemma=True
    )
    
    # Calcul des CAVs (CPU uniquement)
    cavs_concepts = {}
    
    print("🧮 Calcul des CAVs...")
    for concept in tqdm(concept_name_list, desc="CAVs"):
        concept_data = df_with_embeddings.loc[
            df_with_embeddings['consolidated_concepts'].apply(lambda x: concept in x)
        ]
        
        if concept_data.empty:
            print(f"⚠️  Pas d'échantillons pour '{concept}'")
            continue
        
        concept_embeddings = torch.stack([emb for emb in concept_data['embeddings'].values]).cpu()
        
        other_data = df_with_embeddings.loc[
            df_with_embeddings['consolidated_concepts'].apply(lambda x: concept not in x)
        ]
        
        if other_data.empty:
            print(f"⚠️  Pas d'échantillons négatifs pour '{concept}'")
            continue
        
        other_embeddings = torch.stack([emb for emb in other_data['embeddings'].values]).cpu()
        
        cav = concept_embeddings.mean(dim=0) - other_embeddings.mean(dim=0)
        cavs_concepts[concept] = cav.numpy()
        
        print(f"✅ CAV '{concept}' : pos={len(concept_data)}, neg={len(other_data)}")
    
    # Sauvegarde
    _save_cavs_json(cavs_concepts, config, mode='gemma')
    
    return cavs_concepts



def compute_cavs_mean_minus_image(
    loader,
    baseline_model,
    config,
    concept_list=None
):
    """
    Calcule les CAVs pour modalité IMAGE via méthode mean-minus-others.
    """
    print("=" * 60)
    print("🟢 CALCUL CAVs - MODALITÉ IMAGE (mean-minus-others)")
    print("=" * 60)
    
    # Extraction des embeddings
    pos_embeds, neg_embeds, concept_list = _extract_embeddings_from_loader_image(
        loader, baseline_model, config, concept_list
    )

    # Calcul des CAVs
    cavs = {}
    device = config.device
    
    print("🧮 Calcul des CAVs...")
    for concept_name in tqdm(concept_list, desc="CAVs"):
        # ✅ Normalisation avec vérification (comme multimodal)
        if not concept_name.startswith("concept_"):
            concept_name = "concept_" + concept_name
        
        if concept_name not in pos_embeds or concept_name not in neg_embeds:
            print(f"⚠️  Concept '{concept_name}' absent")
            continue
        
        if len(pos_embeds[concept_name]) == 0 or len(neg_embeds[concept_name]) == 0:
            print(f"⚠️  '{concept_name}' : données insuffisantes")
            continue
        
        try:
            pos_tensor = torch.stack(pos_embeds[concept_name]).to(device)
            neg_tensor = torch.stack(neg_embeds[concept_name]).to(device)
            
            cav_vector = (pos_tensor.mean(0) - neg_tensor.mean(0)).cpu().numpy()
            
            clean_name = concept_name.replace('concept_', '')
            cavs[clean_name] = cav_vector
            
            print(f"✅ CAV '{clean_name}' : pos={len(pos_embeds[concept_name])}, neg={len(neg_embeds[concept_name])}")
            
        except Exception as e:
            print(f"❌ Erreur pour '{concept_name}' : {e}")
    
    # Sauvegarde
    _save_cavs_json(cavs, config, mode='image')
    
    return cavs
    

def compute_cavs_mean_minus_multimodal(
    loader,
    baseline_model,
    processor,
    config,
    concept_list=None,
    use_images=True
):
    """
    Calcule les CAVs pour modalité MULTIMODAL via méthode mean-minus-others.
    
    Args:
        loader: DataLoader retournant dict avec 'text', 'image', 'concept_*'
        baseline_model: Modèle multimodal (CLIP/BLIP)
        processor: Processor pour prétraiter text + images
        config: Config avec device, SAVE_PATH, etc.
        concept_list: Liste des concepts (auto-détecté si None)
        use_images: Si True, fusion text+image; sinon text seul
        
    Returns:
        dict: {concept_name: cav_vector (numpy array)}
    """
    print("=" * 60)
    print("🟡 CALCUL CAVs - MODALITÉ MULTIMODAL (mean-minus-others)")
    print("=" * 60)
    
    # Extraction des embeddings
    pos_embeds, neg_embeds, concept_list = _extract_embeddings_from_loader_multimodal(
        loader, baseline_model, processor, config, concept_list, use_images
    )
    
    # Calcul des CAVs
    cavs = {}
    device = config.device
    
    print("🧮 Calcul des CAVs...")
    for cname in tqdm(concept_list, desc="CAVs"):
        # Normalisation nom
        if not cname.startswith("concept_"):
            cname = "concept_" + cname
        
        if cname not in pos_embeds or cname not in neg_embeds:
            print(f"⚠️  Concept '{cname}' absent")
            continue
        
        if len(pos_embeds[cname]) == 0 or len(neg_embeds[cname]) == 0:
            print(f"⚠️  '{cname}' : données insuffisantes")
            continue
        
        try:
            pos = torch.stack(pos_embeds[cname]).to(device)
            neg = torch.stack(neg_embeds[cname]).to(device)
            
            cav_vector = (pos.mean(0) - neg.mean(0)).cpu().numpy()
            
            clean_name = cname.replace('concept_', '')
            cavs[clean_name] = cav_vector
            
            print(f"✅ CAV '{clean_name}' : pos={len(pos_embeds[cname])}, neg={len(neg_embeds[cname])}")
            
        except Exception as e:
            print(f"❌ Erreur pour '{cname}' : {e}")
    
    # Sauvegarde
    _save_cavs_json(cavs, config, mode='multimodal')
    
    return cavs

# ============================================================================
# 4. UTILITAIRES DE SAUVEGARDE
# ============================================================================

def _save_cavs_json(cavs, config, mode='text'):
    """
    Sauvegarde les CAVs au format JSON.
    
    Args:
        cavs: dict {concept: numpy array}
        config: Config avec SAVE_PATH, model_name, cavs_type, annotation
        mode: 'text', 'image', 'multimodal', 'gemma'
    """
    json_save_path = os.path.join(
        config.SAVE_PATH,
        "blue_checkpoints",
        config.model_name,
        "cavs",
        config.cavs_type
    )
    os.makedirs(json_save_path, exist_ok=True)
    
    json_file_path = os.path.join(
        json_save_path,
        f"cavs_mean_{config.annotation}.json"
    )
    
    with open(json_file_path, 'w') as f:
        json.dump({k: v.tolist() for k, v in cavs.items()}, f, indent=4)
    
    print(f"💾 CAVs sauvegardés : {json_file_path}")


def _save_r2_metrics_and_cavs(metrics, cavs, config, save_cavs=True, mode='text'):
    """
    Sauvegarde les métriques R² et les CAVs issus de la régression.
    
    Args:
        metrics: dict de métriques par concept
        cavs: dict {concept: cav_normalized (numpy)}
        config: Config avec SAVE_PATH, model_name, cavs_type, annotation
        save_cavs: Si True, sauvegarde aussi les CAVs en pickle
        mode: 'text', 'image', 'multimodal', 'gemma'
    """
    save_dir = os.path.join(
        config.SAVE_PATH,
        "blue_checkpoints",
        config.model_name,
        "cavs",
        config.cavs_type
    )
    os.makedirs(save_dir, exist_ok=True)
    
    # Métriques JSON
    metrics_file = f"r2_identifiability_{config.cavs_type}_{config.annotation}_{mode}.json"
    metrics_path = os.path.join(save_dir, metrics_file)
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)
    
    print(f"💾 Métriques R² sauvegardées : {metrics_path}")
    
    # CAVs pickle
    if save_cavs:
        cavs_file = f"cavs_regression_{config.cavs_type}_{config.annotation}_{mode}.pkl"
        cavs_path = os.path.join(save_dir, cavs_file)
        
        with open(cavs_path, 'wb') as f:
            pickle.dump(cavs, f)
        
        print(f"💾 CAVs (régression) sauvegardés : {cavs_path}")
