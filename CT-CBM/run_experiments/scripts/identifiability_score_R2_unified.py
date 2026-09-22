"""
Identifiability Score R² - Unified Version
===========================================

Calcul unifié du score d'identifiabilité R² pour les 3 modalités :
- text (BERT, Gemma)
- image (CLIP vision, BLIP vision)
- multimodal (CLIP/BLIP avec texte + images)

Version: 2.0 - Architecture unifiée avec dispatch automatique
Date: 2025-11-21
"""

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from collections import defaultdict
import json
import pickle
import os


# ============================================================================
# 1. FONCTION DISPATCHER PRINCIPALE
# ============================================================================

def compute_r2_identifiability_score(
    df_cosine,
    embedder_model,
    embedder_tokenizer,
    config,
    concept_columns=None,
    r2_cutoff=None,
    test_size=0.3,
    random_state=42,
    save_cavs=True,
    modality_mode='text'
):
    """
    Dispatcher unifié pour le calcul du score R² d'identifiabilité.
    Détecte automatiquement le mode de modalité et route vers la fonction appropriée.
    
    Args:
        df_cosine: DataFrame ou DataLoader selon le mode
        embedder_model: Modèle pour extraire les embeddings
        embedder_tokenizer: Tokenizer (peut être None pour image)
        config: Config avec device, SAVE_PATH, model_name, etc.
        concept_columns: Liste des colonnes de concepts (auto-détecté si None)
        r2_cutoff: Seuil R² pour filtrer les concepts
        test_size: Proportion du set de validation
        random_state: Seed pour reproductibilité
        save_cavs: Si True, sauvegarde les CAVs calculés
        modality_mode: 'text', 'image', 'multimodal'
        
    Returns:
        tuple: (train_df, val_df, metrics, filtered_concepts, cavs)
    """
    # Détection automatique du mode si nécessaire
    # if modality_mode == 'auto':
    #     # modality_mode = _detect_modality_mode(config, df_cosine, embedder_tokenizer)
    #     print("🔍 Mode auto pas encore implémenté")
    modality_mode = config.modality_mode.lower() if hasattr(config, 'modality_mode') else 'text'
    print(f"🔍 Mode de modalité détecté : {modality_mode}")

    # Dispatch vers la fonction appropriée
    if modality_mode == 'text':
        # Détection BERT vs Gemma
        model_type = 'gemma' if 'gemma' in config.model_name.lower() else 'bert'
        return compute_r2_identifiability_score_text(
            df_cosine=df_cosine,
            embedder_model=embedder_model,
            embedder_tokenizer=embedder_tokenizer,
            config=config,
            concept_columns=concept_columns,
            r2_cutoff=r2_cutoff,
            test_size=test_size,
            random_state=random_state,
            save_cavs=save_cavs,
            model_type=model_type
        )
    elif modality_mode == 'image':
        return compute_r2_identifiability_score_image(
            loader=df_cosine,  # C'est un DataLoader ici
            embedder_model=embedder_model,
            config=config,
            concept_columns=concept_columns,
            r2_cutoff=r2_cutoff,
            test_size=test_size,
            random_state=random_state,
            save_cavs=save_cavs
        )
    elif modality_mode == 'multimodal':
        return compute_r2_identifiability_score_multimodal(
            loader=df_cosine,  # C'est un DataLoader ici
            embedder_model=embedder_model,
            processor=embedder_tokenizer,  # Pour multimodal c'est un processor
            config=config,
            concept_columns=concept_columns,
            r2_cutoff=r2_cutoff,
            test_size=test_size,
            random_state=random_state,
            save_cavs=save_cavs
        )
    else:
        raise ValueError(f"Mode non supporté : {modality_mode}")


# def _detect_modality_mode(config, data_source, tokenizer):
#     """
#     Détecte automatiquement le mode de modalité.
    
#     Args:
#         config: Configuration avec model_name
#         data_source: DataFrame ou DataLoader
#         tokenizer: Tokenizer/Processor (peut être None)
        
#     Returns:
#         str: 'text', 'image', ou 'multimodal'
#     """
#     # Détection basée sur le nom du modèle

#     model_name = config.model_name.lower()
    
#     if 'clip' in model_name or 'blip' in model_name:
#         # Vérifier le type de data_source
#         if isinstance(data_source, pd.DataFrame):
#             return 'text'
#         else:
#             # C'est un DataLoader - vérifier le contenu
#             try:
#                 first_batch = next(iter(data_source))
#                 if 'text' not in first_batch or not any(first_batch['text']):
#                     return 'image'
#                 else:
#                     return 'multimodal'
#             except:
#                 return 'multimodal'  # Par défaut
#     else:
#         # BERT, Gemma, etc.
#         return 'text'

# def _detect_modality_mode(config, data_source, tokenizer):
#     """Détecte automatiquement le mode de modalité."""
#     from clip_config import is_clip_family
#     model_name = config.model_name.lower()
    
#     if is_clip_family(model_name):
#         if isinstance(data_source, pd.DataFrame):
#             return 'text'
#         else:
#             try:
#                 first_batch = next(iter(data_source))
#                 if 'text' not in first_batch or not any(first_batch['text']):
#                     return 'image'
#                 else:
#                     return 'multimodal'
#             except:
#                 return 'multimodal'
#     else:
#         return 'text'

# ============================================================================
# 2. EXTRACTION DES EMBEDDINGS (Helpers communs)
# ============================================================================

def _extract_embeddings_from_dataframe(
    df, 
    model,         
    tokenizer, 
    config,
    is_gemma=False
):
    """
    Extrait les embeddings pour chaque texte d'un DataFrame.
    
    Args:
        df: DataFrame contenant au moins une colonne 'text'
        model: Modèle avec méthode get_pooled_output
        tokenizer: Tokenizer associé
        config: Config avec device, max_len
        is_gemma: Si True, utilise get_pooled_output(input_ids) sans attention_mask
        
    Returns:
        pd.DataFrame: DataFrame avec colonne 'embeddings' ajoutée
    """
    device = config.device
    model.to(device).eval()
    
    df_copy = df.copy()
    df_copy['embeddings'] = None
    df_copy['embeddings'] = df_copy['embeddings'].astype(object)
    
    print(f"🔄 Extraction des embeddings (DataFrame, {len(df)} samples)...")
    
    for idx, row in tqdm(df_copy.iterrows(), total=len(df_copy), desc="Embeddings"):
        text = row["text"]
        
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
                outputs = model.get_pooled_output(input_ids)
            else:
                outputs = model.get_pooled_output(input_ids, attention_mask)
        
        # Stockage CPU
        embedding_cpu = outputs.flatten().detach().cpu()
        df_copy.at[idx, 'embeddings'] = embedding_cpu
        
        # Cleanup
        del input_ids, attention_mask, outputs
        torch.cuda.empty_cache()
    
    return df_copy


def _extract_embeddings_from_loader_image(
    loader,
    model,
    config,
    concept_list=None
):
    """
    Extrait les embeddings d'images depuis un DataLoader.
    
    Args:
        loader: DataLoader retournant dict avec 'image' et 'concept_*'
        model: Modèle vision (CLIP/BLIP)
        config: Config avec device
        concept_list: Liste des concepts (détectée auto si None)
        
    Returns:
        tuple: (pos_embeds, neg_embeds, concept_list)
    """
    device = config.device
    model.to(device).eval()
    
    # Détection automatique des concepts
    if concept_list is None:
        first_batch = next(iter(loader))
        concept_list = [k for k in first_batch.keys() 
                       if k.startswith("concept_") or k.startswith("dummy_")]
        
        # Reconstruction du loader
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
            if hasattr(model, 'get_pooled_output'):
                pooled_output = model.get_pooled_output(pixel_values=images)
            elif hasattr(model, 'get_image_features'):
                pooled_output = model.get_image_features(pixel_values=images)
            elif hasattr(model, 'vision_model'):
                pooled_output = model.vision_model(pixel_values=images).pooler_output
            else:
                raise ValueError("Méthode d'extraction embeddings image non reconnue")
        
        pooled_output_cpu = pooled_output.cpu()
        
        # Dispatch pos/neg
        for i in range(batch_size):
            for concept_name in concept_list:
                if concept_name in batch:
                    if batch[concept_name][i] == 1:
                        pos_embeds[concept_name].append(pooled_output_cpu[i])
                    else:
                        neg_embeds[concept_name].append(pooled_output_cpu[i])
        
        torch.cuda.empty_cache()
    
    return pos_embeds, neg_embeds, concept_list


def _extract_embeddings_from_loader_multimodal(
    loader,
    model,
    processor,
    config,
    concept_list=None,
    use_images=True
):
    """
    Extrait les embeddings multimodaux (texte + image) depuis un DataLoader.
    
    Args:
        loader: DataLoader retournant dict avec 'text', 'image', 'concept_*'
        model: Modèle multimodal (CLIP/BLIP)
        processor: Processor pour prétraiter text + images
        config: Config avec device
        concept_list: Liste des concepts (détectée auto si None)
        use_images: Si True, utilise text+image; sinon text seul
        
    Returns:
        tuple: (pos_embeds, neg_embeds, concept_list)
    """
    device = config.device
    model.to(device).eval()
    
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
            pooled_output = model.get_pooled_output(
                batch_inputs.input_ids,
                batch_inputs.attention_mask,
                batch_inputs.pixel_values if use_images else None
            )
        
        pooled_output_cpu = pooled_output.cpu()
        
        # Dispatch pos/neg
        for i in range(len(texts)):
            for cname in concept_list:
                # Normalisation du nom du concept
                if not (cname.startswith("dummy_") or cname.startswith("concept_")):
                    cname = "concept_" + cname
                
                if batch[cname][i] == 1:
                    pos_embeds[cname].append(pooled_output_cpu[i])
                else:
                    neg_embeds[cname].append(pooled_output_cpu[i])
        
        torch.cuda.empty_cache()
    
    return pos_embeds, neg_embeds, concept_list


# ============================================================================
# 3. CALCUL DU SCORE R² - VERSION text
# ============================================================================

def compute_r2_identifiability_score_text(
    df_cosine,
    embedder_model,
    embedder_tokenizer,
    config,
    concept_columns=None,
    r2_cutoff=None,
    test_size=0.3,
    random_state=42,
    save_cavs=True,
    model_type='bert'
):
    """
    Calcule le score R² d'identifiabilité pour mode text.
    
    Args:
        df_cosine: DataFrame avec colonnes 'text', 'label', concept1, concept2, ...
        embedder_model: Modèle pour extraire les embeddings
        embedder_tokenizer: Tokenizer associé
        config: Config avec device, SAVE_PATH, etc.
        concept_columns: Liste des colonnes de concepts (auto-détecté si None)
        r2_cutoff: Seuil R² pour filtrer les concepts
        test_size: Proportion du set de validation
        random_state: Seed pour reproductibilité
        save_cavs: Si True, sauvegarde les CAVs calculés
        model_type: 'bert' ou 'gemma'
        
    Returns:
        tuple: (train_df, val_df, metrics, filtered_concepts, cavs)
    """
    print("=" * 60)
    print(f"🔴 CALCUL SCORE R² IDENTIFIABILITÉ - text ({model_type.upper()})")
    print("=" * 60)
    
    device = config.device
    
    # Identification des colonnes de concepts
    if concept_columns is None:
        concept_columns = [col for col in df_cosine.columns 
                          if col not in ["text", 'label', 'embeddings']]
    
    print(f"📊 {len(concept_columns)} concepts : {concept_columns}")
    
    # Split train/val
    train_df, val_df = train_test_split(
        df_cosine,
        test_size=test_size,
        random_state=random_state,
        stratify=df_cosine['label'] if df_cosine['label'].nunique() <= 10 else None
    )
    
    print(f"📊 Split : Train={len(train_df)}, Val={len(val_df)}")
    
    # Extraction embeddings
    def extract_embeddings_batch(texts, model, tokenizer, device, config, batch_size=32, is_gemma=False):
        model.to(device).eval()
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Embeddings"):
            batch_texts = texts[i:i+batch_size]
            
            with torch.no_grad():
                encoded = tokenizer(
                    batch_texts,
                    return_tensors='pt',
                    truncation=True,
                    padding=True,
                    max_length=config.max_len,
                    return_token_type_ids=False
                )
                
                for k in encoded:
                    encoded[k] = encoded[k].to(device)
                
                # Gestion BERT vs Gemma
                if is_gemma:
                    if hasattr(model, 'get_pooled_output'):
                        batch_emb = model.get_pooled_output(encoded['input_ids'])
                    else:
                        output = model(**encoded)
                        batch_emb = output.last_hidden_state.mean(dim=1)
                else:
                    batch_emb = model.get_pooled_output(**encoded)
                
                embeddings.append(batch_emb.cpu())
                
                del encoded
                torch.cuda.empty_cache()
        
        return torch.cat(embeddings, dim=0).numpy()
    
    print("🔄 Extraction des embeddings...")
    is_gemma = (model_type == 'gemma')
    X_train = extract_embeddings_batch(
        train_df["text"].tolist(), embedder_model, embedder_tokenizer, device, config, is_gemma=is_gemma
    )
    X_val = extract_embeddings_batch(
        val_df["text"].tolist(), embedder_model, embedder_tokenizer, device, config, is_gemma=is_gemma
    )
    
    print(f"📊 Embeddings : Train={X_train.shape}, Val={X_val.shape}")
    
    # Régression + CAVs
    metrics = {}
    cavs = {}
    train_df_copy = train_df.copy()
    val_df_copy = val_df.copy()
    
    print("🔄 Régression linéaire et calcul des CAVs...")
    
    for concept in tqdm(concept_columns, desc="Régression"):
        # Nettoyage du nom du concept
        clean_concept = concept.replace('concept_', '').replace('dummy_', '')
        
        # Targets continus
        y_train = train_df[concept].values
        y_val = val_df[concept].values
        
        # Gestion NaN
        if np.isnan(y_train).any() or np.isnan(y_val).any():
            y_train = np.nan_to_num(y_train)
            y_val = np.nan_to_num(y_val)
        
        # Régression
        reg = LinearRegression()
        reg.fit(X_train, y_train)
        
        # CAV = coefficients normalisés
        cav_raw = reg.coef_
        cav_normalized = cav_raw / np.linalg.norm(cav_raw)
        cavs[clean_concept] = cav_normalized
        
        # Prédictions
        train_preds = reg.predict(X_train)
        val_preds = reg.predict(X_val)
        
        # Métriques
        r2_val = r2_score(y_val, val_preds)
        r2_train = r2_score(y_train, train_preds)
        mse = mean_squared_error(y_val, val_preds)
        mae = mean_absolute_error(y_val, val_preds)
        rmse = np.sqrt(mse)
        
        metrics[clean_concept] = {
            "R2_val": float(r2_val),
            "R2_train": float(r2_train),
            "MSE": float(mse),
            "MAE": float(mae),
            "RMSE": float(rmse),
            "y_true_mean": float(np.mean(y_val)),
            "y_true_std": float(np.std(y_val)),
            "y_pred_mean": float(np.mean(val_preds)),
            "y_pred_std": float(np.std(val_preds)),
            "cav_norm": float(np.linalg.norm(cav_raw)),
            "intercept": float(reg.intercept_)
        }
        
        train_df_copy[f"pred_{clean_concept}"] = train_preds
        val_df_copy[f"pred_{clean_concept}"] = val_preds
    
    # Sauvegarde
    _save_r2_metrics_and_cavs(metrics, cavs, config, save_cavs, mode=model_type)
    
    # Filtrage
    if r2_cutoff is not None:
        filtered_concepts = [c for c, m in metrics.items() if m["R2_val"] >= r2_cutoff]
        print(f"📊 Concepts filtrés (R² ≥ {r2_cutoff}) : {len(filtered_concepts)}/{len(concept_columns)}")
    else:
        filtered_concepts = list(cavs.keys())
    
    # Résumé
    _print_r2_summary(metrics, cavs)
    
    return train_df_copy, val_df_copy, metrics, filtered_concepts, cavs


# ============================================================================
# 4. CALCUL DU SCORE R² - VERSION image
# ============================================================================

def compute_r2_identifiability_score_image(
    loader,
    embedder_model,
    config,
    concept_columns=None,
    r2_cutoff=None,
    test_size=0.3,
    random_state=42,
    save_cavs=True
):
    """
    Calcule le score R² d'identifiabilité pour mode image.
    VERSION CORRIGÉE : compte correctement le nombre d'exemples
    """
    print("=" * 60)
    print("🔴 CALCUL SCORE R² IDENTIFIABILITÉ - image")
    print("=" * 60)
    
    device = config.device
    embedder_model.to(device).eval()
    
    # Détection automatique des concepts
    if concept_columns is None:
        first_batch = next(iter(loader))
        concept_list = [k for k in first_batch.keys() 
                       if k.startswith("concept_") or k.startswith("dummy_")]
        
        # Reconstruction du loader
        loader = torch.utils.data.DataLoader(
            loader.dataset, 
            batch_size=loader.batch_size,
            shuffle=False, 
            num_workers=getattr(loader, 'num_workers', 0),
            pin_memory=True
        )
        print(f"📝 Concepts détectés : {len(concept_list)}")
    else:
        concept_list = concept_columns
    
    # ============================================================================
    # EXTRACTION DES EMBEDDINGS - VERSION CORRIGÉE
    # ============================================================================
    all_embeddings = []
    concept_values = {c: [] for c in concept_list}
    all_labels = []
    
    print(f"🔄 Extraction des embeddings (Image DataLoader, {len(loader)} batches)...")
    
    from tqdm import tqdm
    import numpy as np
    
    for batch in tqdm(loader, desc="Image embeddings"):
        images = batch["image"].to(device)
        labels = batch["label"]
        
        with torch.no_grad():
            # Extraction des features image
            if hasattr(embedder_model, 'get_image_features'):
                embeddings = embedder_model.get_image_features(pixel_values=images)
            elif hasattr(embedder_model, 'get_pooled_output'):
                embeddings = embedder_model.get_pooled_output(pixel_values=images)
            elif hasattr(embedder_model, 'vision_model'):
                embeddings = embedder_model.vision_model(pixel_values=images).pooler_output
            else:
                raise ValueError("Méthode d'extraction embeddings image non reconnue")
            
            embeddings = embeddings.cpu().numpy()
        
        # FIX CRITIQUE : Déterminer le batch_size réel
        if embeddings.ndim == 1:
            batch_size = 1
            embeddings = embeddings.reshape(1, -1)
        else:
            batch_size = embeddings.shape[0]
        
        # Append les embeddings (chaque ligne = 1 exemple)
        all_embeddings.append(embeddings)
        
        # Pour chaque concept
        for c in concept_list:
            if c in batch:
                vals = batch[c].cpu().numpy() if isinstance(batch[c], torch.Tensor) else batch[c]
                # S'assurer que c'est un array 1D de longueur batch_size
                if isinstance(vals, (int, float)):
                    vals = np.array([float(vals)] * batch_size)
                elif vals.size == 1:
                    vals = np.array([float(vals)] * batch_size)
                else:
                    vals = vals.flatten()[:batch_size]
                
                concept_values[c].extend(vals.tolist())
            else:
                concept_values[c].extend([0.0] * batch_size)
        
        # Labels
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        
        if np.isscalar(labels) or labels.size == 1:
            all_labels.extend([int(labels)] * batch_size)
        else:
            all_labels.extend(labels.flatten()[:batch_size].tolist())
        
        torch.cuda.empty_cache()
    
    # Vérification et assemblage
    X = np.vstack(all_embeddings)
    
    print(f"✓ Embeddings extraits : {X.shape}")
    print(f"✓ Nombre d'exemples : {len(all_labels)}")
    
    # ASSERTIONS pour détecter le bug
    assert X.shape[0] == len(all_labels), \
        f"❌ MISMATCH : {X.shape[0]} embeddings vs {len(all_labels)} labels"
    
    for c, vals in concept_values.items():
        assert len(vals) == len(all_labels), \
            f"❌ MISMATCH concept {c} : {len(vals)} valeurs vs {len(all_labels)} labels"
    
    # Création DataFrame
    import pandas as pd
    df = pd.DataFrame(concept_values)
    df['label'] = all_labels
    # ============================================================================
    
    # Split train/val
    from sklearn.model_selection import train_test_split
    train_idx, val_idx = train_test_split(
        np.arange(len(df)),
        test_size=test_size,
        random_state=random_state
    )
    
    X_train, X_val = X[train_idx], X[val_idx]
    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()
    
    print(f"📊 Split : Train={len(train_df)}, Val={len(val_df)}")
    
    # Régression + CAVs
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    
    metrics = {}
    cavs = {}
    
    print("🔄 Régression linéaire et calcul des CAVs...")
    
    for concept in tqdm(concept_list, desc="Régression"):
        clean_concept = concept.replace('concept_', '').replace('dummy_', '')
        
        y_train = train_df[concept].values
        y_val = val_df[concept].values
        
        # Régression
        reg = LinearRegression()
        reg.fit(X_train, y_train)
        
        # CAV
        cav_raw = reg.coef_
        cav_normalized = cav_raw / np.linalg.norm(cav_raw)
        cavs[clean_concept] = cav_normalized
        
        # Prédictions
        train_preds = reg.predict(X_train)
        val_preds = reg.predict(X_val)
        
        # Métriques
        r2_val = r2_score(y_val, val_preds)
        r2_train = r2_score(y_train, train_preds)
        
        metrics[clean_concept] = {
            "R2_val": float(r2_val),
            "R2_train": float(r2_train),
            "MSE": float(mean_squared_error(y_val, val_preds)),
            "MAE": float(mean_absolute_error(y_val, val_preds)),
            "RMSE": float(np.sqrt(mean_squared_error(y_val, val_preds)))
        }
        
        train_df[f"pred_{clean_concept}"] = train_preds
        val_df[f"pred_{clean_concept}"] = val_preds
    
    # Sauvegarde
    _save_r2_metrics_and_cavs(metrics, cavs, config, save_cavs, mode='image')
    
    # Filtrage
    if r2_cutoff is not None:
        filtered_concepts = [c for c, m in metrics.items() if m["R2_val"] >= r2_cutoff]
    else:
        filtered_concepts = list(cavs.keys())
    
    _print_r2_summary(metrics, cavs)
    
    return train_df, val_df, metrics, filtered_concepts, cavs

# ============================================================================
# 5. CALCUL DU SCORE R² - VERSION MULTIMODAL
# ============================================================================

# REMPLACEZ LA FONCTION compute_r2_identifiability_score_multimodal
# DANS identifiability_score_R2_unified.py

def compute_r2_identifiability_score_multimodal(
    loader,
    embedder_model,
    processor,
    config,
    concept_columns=None,
    r2_cutoff=None,
    test_size=0.3,
    random_state=42,
    save_cavs=True,
    use_images=True
):
    """
    Calcule le score R² d'identifiabilité pour mode MULTIMODAL.
    VERSION CORRIGÉE : compte correctement le nombre d'exemples
    """
    print("=" * 60)
    print("🔴 CALCUL SCORE R² IDENTIFIABILITÉ - MULTIMODAL")
    print("=" * 60)
    
    device = config.device
    embedder_model.to(device).eval()
    
    # Détection automatique des concepts
    if concept_columns is None:
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
        print(f"📝 Concepts détectés : {len(concept_list)}")
    else:
        concept_list = concept_columns
    
    # ============================================================================
    # EXTRACTION DES EMBEDDINGS - VERSION CORRIGÉE
    # ============================================================================
    all_embeddings = []
    concept_values = {c: [] for c in concept_list}
    all_labels = []
    
    print(f"🔄 Extraction des embeddings (Multimodal DataLoader, {len(loader)} batches)...")
    
    from tqdm import tqdm
    import numpy as np
    
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
            if use_images and hasattr(batch_inputs, 'pixel_values'):
                pooled_output = embedder_model.get_pooled_output(
                    batch_inputs.input_ids,
                    batch_inputs.attention_mask,
                    batch_inputs.pixel_values
                )
            else:
                pooled_output = embedder_model.get_pooled_output(
                    batch_inputs.input_ids,
                    batch_inputs.attention_mask
                )
            
            embeddings = pooled_output.cpu().numpy()
        
        # FIX CRITIQUE : Déterminer le batch_size réel
        if embeddings.ndim == 1:
            batch_size = 1
            embeddings = embeddings.reshape(1, -1)
        else:
            batch_size = embeddings.shape[0]
        
        # Append les embeddings
        all_embeddings.append(embeddings)
        
        # Pour chaque concept
        for c in concept_list:
            # Normalisation du nom du concept
            concept_key = c
            if not (c.startswith("dummy_") or c.startswith("concept_")):
                concept_key = "concept_" + c
            
            if concept_key in batch:
                vals = batch[concept_key].cpu().numpy() if isinstance(batch[concept_key], torch.Tensor) else batch[concept_key]
                
                # S'assurer que c'est un array 1D de longueur batch_size
                if isinstance(vals, (int, float)):
                    vals = np.array([float(vals)] * batch_size)
                elif vals.size == 1:
                    vals = np.array([float(vals)] * batch_size)
                else:
                    vals = vals.flatten()[:batch_size]
                
                concept_values[c].extend(vals.tolist())
            else:
                concept_values[c].extend([0.0] * batch_size)
        
        # Labels
        labels = batch["label"]
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        
        if np.isscalar(labels) or labels.size == 1:
            all_labels.extend([int(labels)] * batch_size)
        else:
            all_labels.extend(labels.flatten()[:batch_size].tolist())
        
        # Cleanup
        del batch_inputs, pooled_output
        torch.cuda.empty_cache()
    
    # Vérification et assemblage
    X = np.vstack(all_embeddings)
    
    print(f"✓ Embeddings extraits : {X.shape}")
    print(f"✓ Nombre d'exemples : {len(all_labels)}")
    
    # ASSERTIONS pour détecter le bug
    assert X.shape[0] == len(all_labels), \
        f"❌ MISMATCH : {X.shape[0]} embeddings vs {len(all_labels)} labels"
    
    for c, vals in concept_values.items():
        assert len(vals) == len(all_labels), \
            f"❌ MISMATCH concept {c} : {len(vals)} valeurs vs {len(all_labels)} labels"
    
    # Création DataFrame
    import pandas as pd
    df = pd.DataFrame(concept_values)
    df['label'] = all_labels
    # ============================================================================
    
    # Split train/val
    from sklearn.model_selection import train_test_split
    train_idx, val_idx = train_test_split(
        np.arange(len(df)),
        test_size=test_size,
        random_state=random_state
    )
    
    X_train, X_val = X[train_idx], X[val_idx]
    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()
    
    print(f"📊 Split : Train={len(train_df)}, Val={len(val_df)}")
    
    # Régression + CAVs
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    
    metrics = {}
    cavs = {}
    
    print("🔄 Régression linéaire et calcul des CAVs...")
    
    for concept in tqdm(concept_list, desc="Régression"):
        clean_concept = concept.replace('concept_', '').replace('dummy_', '')
        
        y_train = train_df[concept].values
        y_val = val_df[concept].values
        
        # Régression
        reg = LinearRegression()
        reg.fit(X_train, y_train)
        
        # CAV
        cav_raw = reg.coef_
        cav_normalized = cav_raw / np.linalg.norm(cav_raw)
        cavs[clean_concept] = cav_normalized
        
        # Prédictions
        train_preds = reg.predict(X_train)
        val_preds = reg.predict(X_val)
        
        # Métriques
        r2_val = r2_score(y_val, val_preds)
        r2_train = r2_score(y_train, train_preds)
        
        metrics[clean_concept] = {
            "R2_val": float(r2_val),
            "R2_train": float(r2_train),
            "MSE": float(mean_squared_error(y_val, val_preds)),
            "MAE": float(mean_absolute_error(y_val, val_preds)),
            "RMSE": float(np.sqrt(mean_squared_error(y_val, val_preds)))
        }
        
        train_df[f"pred_{clean_concept}"] = train_preds
        val_df[f"pred_{clean_concept}"] = val_preds
    
    # Sauvegarde
    _save_r2_metrics_and_cavs(metrics, cavs, config, save_cavs, mode='multimodal')
    
    # Filtrage
    if r2_cutoff is not None:
        filtered_concepts = [c for c, m in metrics.items() if m["R2_val"] >= r2_cutoff]
    else:
        filtered_concepts = list(cavs.keys())
    
    _print_r2_summary(metrics, cavs)
    
    return train_df, val_df, metrics, filtered_concepts, cavs



# ============================================================================
# 6. UTILITAIRES DE SAUVEGARDE ET AFFICHAGE
# ============================================================================

def _save_r2_metrics_and_cavs(metrics, cavs, config, save_cavs=True, mode='text'):
    """
    Sauvegarde les métriques R² et les CAVs issus de la régression.
    
    Args:
        metrics: dict de métriques par concept
        cavs: dict {concept: cav_normalized (numpy)}
        config: Config avec SAVE_PATH, model_name, cavs_type, annotation
        save_cavs: Si True, sauvegarde aussi les CAVs en JSON
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
    metrics_file = f"r2_identifiability_{config.annotation}.json"
    metrics_path = os.path.join(save_dir, metrics_file)
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)
    
    print(f"💾 Métriques R² sauvegardées : {metrics_path}")
    
    # CAVs JSON (conversion numpy -> list)
    if save_cavs:
        cavs_file = f"cavs_{config.cavs_type}_{config.annotation}.json"
        cavs_path = os.path.join(save_dir, cavs_file)
        
        # Conversion des arrays NumPy en listes pour JSON
        cavs_serializable = {
            concept: cav.tolist() if isinstance(cav, np.ndarray) else cav
            for concept, cav in cavs.items()
        }
        
        with open(cavs_path, 'w') as f:
            json.dump(cavs_serializable, f, ensure_ascii=False, indent=4)
        
        print(f"💾 CAVs (régression) sauvegardés : {cavs_path}")

def _print_r2_summary(metrics, cavs):
    """Affiche un résumé des performances"""
    print("\n📊 RÉSUMÉ DES PERFORMANCES:")
    print("-" * 60)
    
    avg_r2 = np.mean([m["R2_val"] for m in metrics.values()])
    print(f"R² moyen (validation) : {avg_r2:.3f}")
    
    best = max(metrics.items(), key=lambda x: x[1]["R2_val"])
    worst = min(metrics.items(), key=lambda x: x[1]["R2_val"])
    
    print(f"Meilleur : {best[0]} (R² = {best[1]['R2_val']:.3f})")
    print(f"Pire : {worst[0]} (R² = {worst[1]['R2_val']:.3f})")
    print(f"🎯 {len(cavs)} CAVs calculés")
    
    if cavs:
        first_cav = list(cavs.values())[0]
        print(f"Dimension des CAVs : {first_cav.shape}")