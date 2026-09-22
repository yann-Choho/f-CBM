import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import os, pickle, json, gc

from torch.cuda.amp import autocast
from tqdm.auto import tqdm



def compute_attributions_from_dataloader(dataloader, model, tokenizer, lig, device):
    """
    Calcule les attributions sur GPU/CPU pour chaque batch du dataloader.
    """
    attributions = []
    print("Début du calcul des attributions…")

    model_dtype = torch.float32 #next(lig.model.parameters()).dtype  # dtype du modèle

    for batch in tqdm(dataloader, desc="Attributions", unit="batch"):

        # 3) Préparer un batch
        texts  = batch["text"]
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        
        # 4) Tokenisation
        inputs = tokenizer(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            #do_rescale=False
        )
        
        # 5) Envoyer et caster
        model_dtype = next(model.parameters()).dtype  # float32 ou float16
        
        input_ids      = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        pixel_values   = inputs["pixel_values"].to(device).to(model_dtype)
        
        # 6) Calculer pooled_output (float32) ET activer requires_grad
        pooled_output = model.get_pooled_output(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values
        )
        # Assurez-vous que c’est float32 et que les gradients sont activés
        pooled_output = pooled_output.to(torch.float32)
        pooled_output.requires_grad_(True)

        # 7) Définir la baseline : un vecteur zéro de même forme
        baseline = torch.zeros_like(pooled_output)

        # 8) Lancer l’attribution
        ctx = torch.autocast(device_type="cuda", dtype=torch.float32) if device.type == "cuda" else torch.no_grad()
        with torch.enable_grad(), ctx:
            attr, delta = lig.attribute(
                inputs=pooled_output,            # Tensor [batch,1024], requires_grad=True
                baselines=baseline,              # même shape
                target=labels,                   # labels pour chaque échantillon
                n_steps=50,
                internal_batch_size=1,
                return_convergence_delta=True
            )

        # Sauvegarde attributs
        for a in attr:
            attributions.append(a.detach().cpu())

        # Libération mémoire
        del inputs, pooled_output, baseline, attr
        torch.cuda.empty_cache()
        gc.collect()

    print("Attributions terminées !")
    return pd.DataFrame({"attributions": attributions})




###############################################
# Fonction 2 : Calcul des similarités cosinus GPU
###############################################
def compute_similarity_cav_text(attribution, cav, device):
    """
    Calcule la similarité cosinus entre les attributions et un vecteur CAV.
    Ici, attribution est supposé être de forme (batch, seq_len, dim) et
    on prend la première position de la séquence pour le calcul (à adapter si nécessaire).
    
    Args:
        attribution (torch.Tensor): Tenseur d'attributions (sur le GPU).
        cav (torch.Tensor): Vecteur CAV déjà converti et placé sur le device.
        device: Device (pour être sûr que tout est sur le même device).
    
    Returns:
        torch.Tensor: Tenseur de similarité cosinus (1D, taille = batch).
    """
    # On s'assure que cav a la dimension (1, dim) pour le broadcasting
    cav = cav.unsqueeze(0).to(device)
    # On suppose que l'attribution a forme (batch, seq_len, dim) et on choisit la première position (c'est à dire le CLS token ici)
    # Vous pouvez adapter cette étape selon votre cas.

    # att_first = attribution[:, 0, :].to(device)
    att_first = attribution.to(device)  # CLS token /token EOS déjà selectionné dans al sortie du CLIP

    # Calcul de la similarité cosinus sur la dimension des features
    similarity = F.cosine_similarity(cav, att_first, dim=1)
    return similarity

def compute_cosine_similarities(attributions_df, df, cavs, device):
    """
    Pour chaque ligne de attributions_df, calcule la similarité cosinus entre l'attribution et
    chacun des vecteurs de concepts présents dans 'cavs'. Le résultat est ajouté aux colonnes
    correspondantes dans df.
    
    Args:
        attributions_df (pd.DataFrame): DataFrame contenant une colonne 'attributions' avec des tenseurs.
        df (pd.DataFrame): DataFrame d'origine auquel on va ajouter les colonnes de similarités.
        cavs (dict): Dictionnaire de vecteurs CAV (les tenseurs doivent être sur le même device).
        device: Device (ex: torch.device("cuda")).
    
    Returns:
        pd.DataFrame: DataFrame mis à jour avec les colonnes 'cos_{concept}'.
    """
    # Initialisation des colonnes de similarités dans df
    for concept in cavs.keys():
        df[f'cos_{concept}'] = 0.0  # valeur par défaut
    
    print("Début du calcul des similarités cosinus...")
    for i in tqdm(range(attributions_df.shape[0]), desc='Cosinus', unit='row'):
        # Charger l'attribution du batch (ici, on suppose qu'on a traité chaque texte séparément)
        # Pour optimiser, vous pourriez regrouper les calculs en batch.
        attrib = attributions_df.at[i, 'attributions'].to(device)  # Convertir sur GPU
        
        # Pour chaque concept, calculer et stocker la similarité
        for concept, cav in cavs.items():
            sim = compute_similarity_cav_text(attrib.unsqueeze(0), cav, device)  # attrib.unsqueeze(0) pour avoir un batch de taille 1
            df.at[i, f'cos_{concept}'] = sim.item()
        
        # Libération de la mémoire GPU (optionnel)
        del attrib
        torch.cuda.empty_cache()
        gc.collect()
    print("Calcul des similarités cosinus terminé !")
    return df

###############################################
# Fonction 3 : Post-traitement des similarités cosinus
###############################################
def postprocess_cosine(df, cavs_keys, mode="abs", agg_scope="all"):
    """
    Applique le post-traitement sur les colonnes de similarités cosinus.
    
    Le traitement s'effectue en deux étapes :
    
    1. Transformation des valeurs négatives :
       - "abs"  : on prend la valeur absolue de chaque score.
       - "clip" : on remplace les valeurs négatives par 0.
       
    2. Agrégation des scores cosinus pour chaque concept selon le scope :
       - "all"     : moyenne sur toutes les lignes.
       - "present" : moyenne uniquement sur les lignes où le concept est présent 
                     (défini par df[concept] == 1, où 'concept' est le nom du concept).
    
    Args:
        df (pd.DataFrame): DataFrame contenant les colonnes 'cos_{concept}' ainsi que
                           éventuellement des colonnes nommées par chaque concept (valeurs 0 ou 1).
        cavs_keys (list): Liste des concepts (clés).
        mode (str): Méthode de traitement des valeurs négatives ("abs" ou "clip"). Par défaut "abs".
        agg_scope (str): Mode d'agrégation, "all" pour toutes les lignes ou "present" pour uniquement les lignes où le concept est présent.
        
    Returns:
        tuple: (df mis à jour, sorted_concepts) où sorted_concepts est une liste triée de tuples (colonne_cosinus, moyenne)
    """
    # Liste des colonnes de similarité
    cosine_columns = [f'cos_{concept}' for concept in cavs_keys]
    
    # Traitement des valeurs négatives
    if mode == "abs":
        df[cosine_columns] = df[cosine_columns].abs()
    elif mode == "clip":
        df[cosine_columns] = df[cosine_columns].clip(lower=0)
    else:
        raise ValueError("Mode inconnu. Choisir 'abs' ou 'clip'.")
    
    # Agrégation des scores cosinus
    if agg_scope == "present":
        # Pour chaque concept, on ne considère que les lignes où df[concept] == 1
        concept_means = {}
        for concept in cavs_keys:
            col_cos = f'cos_{concept}'

            # On vérifie que le DataFrame contient une colonne indiquant la présence du concept
            if concept in df.columns:
                # Sélection des lignes où le concept est présent (df[concept] == 1)
                present_vals = df.loc[df[concept] == 1, col_cos]
            else:
                # Si la colonne de présence n'existe pas, on utilise toutes les valeurs
                present_vals = df[col_cos]
            concept_means[col_cos] = present_vals.mean() if not present_vals.empty else 0.0
    elif agg_scope == "all":
        concept_means = df[cosine_columns].mean(axis=0)
    else:
        raise ValueError("agg_scope doit être 'all' ou 'present'.")
    
    
    # Tri des concepts par moyenne décroissante
    sorted_concepts = sorted(concept_means.items(), key=lambda x: x[1], reverse=True)

    # Fonction pour nettoyer les noms des concepts
    def clean_concept_name(name):
        import re
        name = name.replace("cos_", "").replace("dummy_ ", "").strip()
        name = re.sub(r"Let me know.*", "", name)  # Supprime les phrases inutiles
        return " ".join(name.split())  # Réduit les espaces multiples

    # Appliquer le nettoyage sur la liste chargée
    sorted_concepts = [(clean_concept_name(name), score) for name, score in sorted_concepts] # expérimental
    
    return df, sorted_concepts


# def forward_LIG_black_box(input_ids, attention_mask=None):
#     """
#     Doit renvoyer les logits pour Captum.
#     """
#     outputs = black_box_model.embedder_model(input_ids=input_ids, attention_mask=attention_mask)
#     # on récupère le dernier token (<CLS>) comme pooling
#     pooled = outputs[0][:, -1, :]
#     # print(pooled.shape)
#     logits = black_box_model.classifier(pooled)
#     return logits


