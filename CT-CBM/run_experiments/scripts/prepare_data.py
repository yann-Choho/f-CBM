import os
import json
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ============================================================================
# ENCODAGE DES LABELS
# ============================================================================

def encode_labels(df: pd.DataFrame, save_dir: str, annotation: str):
    """
    Si 'label' est textuel :
      - charge label_dict_<annotation>.json s'il existe, sinon le crée (à partir du DF passé)
      - filtre les labels inconnus vis-à-vis du dictionnaire
      - map texte -> int
    Si 'label' est déjà numérique, ne fait rien.
    Retourne: df, label_to_id (ou None si rien à faire)
    """
    if "label" not in df.columns:
        raise KeyError(" Colonne 'label' absente.")

    dict_path = os.path.join(save_dir, f"label_dict_{annotation}.json")
    os.makedirs(save_dir, exist_ok=True)

    # Rien à faire si déjà numérique
    if pd.api.types.is_numeric_dtype(df["label"]):
        print(" 'label' déjà numérique, encodage sauté.")
        return df, None

    # Nettoyage labels
    df["label"] = df["label"].astype(str).str.strip()

    # Charger ou créer le mapping
    if os.path.exists(dict_path):
        with open(dict_path, "r", encoding="utf-8") as f:
            label_to_id = json.load(f)
        print(f" Dictionnaire existant chargé : {dict_path}")
    else:
        unique_labels = sorted(df["label"].dropna().unique())
        label_to_id = {label: i for i, label in enumerate(unique_labels)}
        with open(dict_path, "w", encoding="utf-8") as f:
            json.dump(label_to_id, f, ensure_ascii=False, indent=4)
        print(f" Nouveau dictionnaire créé : {dict_path}")

    # Filtrer inconnus (et informer)
    before = len(df)
    df = df[df["label"].isin(label_to_id.keys())].copy()
    dropped = before - len(df)
    if dropped:
        print(f" {dropped} lignes supprimées (labels hors dictionnaire).")

    # Mapper texte -> int
    df["label"] = df["label"].map(label_to_id)

    # Sanity check
    if df["label"].isna().any():
        na_cnt = int(df["label"].isna().sum())
        raise ValueError(f" {na_cnt} labels n'ont pas été mappés — incohérence dictionnaire.")

    return df, label_to_id


def _read_csv_if_exists(path: str):
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"✅ CSV chargé: {os.path.basename(path)} ({df.shape[0]} lignes)")
        return df
    else:
        print(f"⚠️ Fichier non trouvé (ignoré): {path}")
        return None


def load_from_csv(annotation, config, require_val: bool = False):
    """
    Charge train / val (optionnel) / test depuis:
      - {config.path_to_input_csv}/train_df_{annotation}.csv
      - {config.path_to_input_csv}/val_df_{annotation}.csv  (si présent, ou exigé si require_val=True)
      - {config.path_to_input_csv}/test_df_{annotation}.csv

    Nettoie 'text'. 
    Encode les labels texte via un dictionnaire unique (créé à partir du train s'il n'existe pas),
    puis applique le même mapping à val et test (labels inconnus filtrés).

    Retourne: (df_train, df_val, df_test)  avec df_val = None si absent et require_val=False.
    """
    base = config.path_to_input_csv
    train_path = os.path.join(base, f"train_df_{annotation}.csv")
    val_path   = os.path.join(base, f"val_df_{annotation}.csv")
    test_path  = os.path.join(base, f"test_df_{annotation}.csv")

    # --- Existence des splits obligatoires ---
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Fichier introuvable: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Fichier introuvable: {test_path}")
    if require_val and not os.path.exists(val_path):
        raise FileNotFoundError(f"Fichier 'val' exigé mais introuvable: {val_path}")

    # --- Lecture ---
    df_train = _read_csv_if_exists(train_path)  # garanti non None
    df_val   = _read_csv_if_exists(val_path)    # peut être None
    df_test  = _read_csv_if_exists(test_path)   # garanti non None

    # --- Nettoyage texte ---
    for df in (df_train, df_val, df_test):
        if df is not None and "text" in df.columns:
            df["text"] = df["text"].astype(str).str.strip()

    # --- Encodage labels : TRAIN (crée/charge le dict si nécessaire) ---
    df_train, _ = encode_labels(
        df=df_train,
        save_dir=base,
        annotation=annotation
    )

    # --- Encodage labels : VAL (réutilise le même dictionnaire) ---
    if df_val is not None:
        df_val, _ = encode_labels(
            df=df_val,
            save_dir=base,
            annotation=annotation
        )

    # --- Encodage labels : TEST (réutilise le même dictionnaire) ---
    df_test, _ = encode_labels(
        df=df_test,
        save_dir=base,
        annotation=annotation
    )

    return df_train, df_val, df_test


# ============================================================================
# UTILITAIRES CONCEPTS & IMAGES
# ============================================================================

def _is_binary_series(s: pd.Series) -> bool:
    """Vérifie si une série ne contient que 0/1 (en ignorant NaN)."""
    vals = pd.unique(s.dropna())
    return set(vals).issubset({0, 1})


def _are_concepts_binary(df: pd.DataFrame, concept_cols: List[str]) -> bool:
    """Vrai si toutes les colonnes concepts sont binaires (0/1)."""
    if not concept_cols:
        return False
    return all(_is_binary_series(df[c]) for c in concept_cols)


def _safe_image_open(path: str):
    with Image.open(path) as img:
        return img.convert("RGB")


class ConvertToRGB:
    def __call__(self, image):
        return image.convert('RGB') if image.mode != 'RGB' else image


# ============================================================================
# FILTRAGE DES CONCEPTS PROBLÉMATIQUES (SUR TRAIN)
# ============================================================================

def filter_problematic_concepts(df: pd.DataFrame, concept_cols: List[str], verbose: bool = True) -> List[str]:
    """
    Retourne uniquement les colonnes qui ne sont PAS entièrement à zéro
    et qui ont au moins une valeur valide.
    (Filtrage à appliquer une fois sur le TRAIN.)
    """
    if not concept_cols:
        return []
    
    valid_concepts = []
    filtered_concepts = []
    
    if verbose:
        print(f"\n🔍 Filtrage de {len(concept_cols)} concepts...")
    
    for concept_col in concept_cols:
        values = pd.to_numeric(df[concept_col], errors="coerce")  # convertit en float, NaN si problème
        valid_values = values.dropna()
        
        if len(valid_values) == 0:
            if verbose:
                print(f"   ❌ '{concept_col}' : aucune valeur valide")
            filtered_concepts.append(concept_col)
            continue
        
        if (valid_values == 0).all():
            if verbose:
                print(f"   ❌ '{concept_col}' : TOUT À ZÉRO")
            filtered_concepts.append(concept_col)
            continue
        
        # sinon, elle est valide
        valid_concepts.append(concept_col)
    
    if verbose:
        print(f"\n✓ {len(valid_concepts)}/{len(concept_cols)} concepts valides")
        if filtered_concepts:
            print(f"❌ Concepts filtrés : {filtered_concepts}\n")
    
    return valid_concepts


# ============================================================================
# DATASET GÉNÉRIQUE MULTIMODAL
# ============================================================================

class GenericMultimodalDataset(Dataset):
    """
    Dataset générique pour texte / image / multimodal.
    - Les concepts = toutes les colonnes préfixées par 'concept_'.
    - Pour 'cb_llm' (concepts continus), on ne fait PAS de ranking par fréquence.
    - Pour 'C3M' / 'our_annotation' (concepts binaires), ranking par classe possible.
    - IMPORTANT : on peut lui passer une base de colonnes de concepts (concept_cols_base),
      typiquement définie sur le TRAIN, afin de forcer la même base sur VAL/TEST.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        config: Any,
        annotation: str,
        modality_mode: str = "multimodal",
        max_len: int = 512,
        transform=None,
        select_most_frequent: Optional[int] = None,
        class_list: Optional[List[int]] = None,
        concept_cols_base: Optional[List[str]] = None,  # 👈 base définie sur le train
    ):
        """
        Args:
            df: DataFrame de travail (déjà encodé via encode_labels / load_from_csv).
            config: objet config avec au minimum:
                - path_to_input_images (str)
                - modality_mode in {"text","image","multimodal"}
            annotation: "C3M", "our_annotation", "cb_llm"
            modality_mode: "text" | "image" | "multimodal"
            select_most_frequent: si not None et concepts binaires -> top-k concepts par classe (SUR TRAIN)
            class_list: filtre optionnel sur certaines classes (entiers après encodage)
            concept_cols_base: liste de concepts imposée (typiquement depuis le TRAIN).
        """
        self.df = df.copy()
        self.config = config
        self.annotation = annotation
        self.modality_mode = modality_mode
        self.max_len = max_len
        self.transform = transform

        # ==================================================================
        # Chargement tokenizer pour le mode texte
        # ==================================================================
        self.embedder_tokenizer = self._load_tokenizer()

        # Filtrage éventuel de classes
        if class_list is not None:
            self.df = self.df[self.df["label"].isin(class_list)].copy()

        # ----------------------------------------------------------------------
        # DÉTERMINATION DES COLONNES CONCEPTS
        # ----------------------------------------------------------------------
        if concept_cols_base is not None:
            # Cas VAL / TEST : on impose la base calculée sur le TRAIN
            self.concept_cols = [c for c in concept_cols_base if c in self.df.columns]
        else:
            # Cas TRAIN : on découvre et filtre les concepts
            self.concept_cols = [c for c in self.df.columns if c.startswith("concept_")]

            if self.concept_cols:
                # Filtrage automatique (zéro partout / aucun signal) uniquement sur le TRAIN
                self.concept_cols = filter_problematic_concepts(
                    df=self.df,
                    concept_cols=self.concept_cols,
                    verbose=True
                )

                # Sélection des concepts les plus fréquents par classe (uniquement si binaire & non cb_llm)
                can_rank = (select_most_frequent is not None) and (self.annotation.lower() != "cb_llm")
                if can_rank and _are_concepts_binary(self.df, self.concept_cols):
                    new_cols: List[str] = []
                    for cls in sorted(self.df["label"].unique().tolist()):
                        sub = self.df[self.df["label"] == cls]
                        # Somme par concept → tri décroissant
                        ranked = sub[self.concept_cols].sum(axis=0).sort_values(ascending=False)
                        # Retirer les déjà sélectionnés pour éviter doublons
                        ranked = ranked[[c for c in ranked.index if c not in new_cols]]
                        new_cols += ranked.head(select_most_frequent).index.tolist()
                    # Conserver uniquement les colonnes retenues
                    self.concept_cols = new_cols

                # Optionnel : supprimer les colonnes concepts "mortes".
                # - Pour 0/1 : somme == 0
                # - Pour continus : variance ~ 0
                if self.concept_cols:
                    if _are_concepts_binary(self.df, self.concept_cols):
                        keep = [c for c in self.concept_cols if self.df[c].sum() > 0]
                    else:
                        # seuil de variance min très faible, évite colonnes constantes
                        keep = [c for c in self.concept_cols if self.df[c].var(ddof=0) > 1e-8]
                    self.concept_cols = keep

        # Harmonisation du chemin d'image selon modality_mode
        # if self.modality_mode in {"image", "multimodal"}:
        #     if "path_to_image" in self.df.columns:
        #         # self.df["path_to_image"] = self.df["path_to_image"].astype(str).apply(
        #         #     lambda x: os.path.join(self.config.path_to_input_images, x)
        #         # )
        #         self.df["path_to_image"] = self.df["path_to_image"].astype(str).apply(
        #             lambda x: os.path.join(self.config.path_to_input_images, os.path.basename(x))
        #         )
        #     else:
        #         raise KeyError("Aucune colonne image trouvée (attendu 'path_to_image').")
        
        # Harmonisation du chemin d'image selon modality_mode
        if self.modality_mode in {"image", "multimodal"}:
            if "path_to_image" in self.df.columns:
                self.df["path_to_image"] = self.df["path_to_image"].astype(str).apply(
                    lambda x: x if os.path.exists(x) else os.path.join(self.config.path_to_input_images, os.path.basename(x))
                )
            else:
                raise KeyError("Aucune colonne image trouvée (attendu 'path_to_image').")

        
        # Vérifs rapides
        if self.modality_mode in {"text", "multimodal"} and "text" not in self.df.columns:
            raise KeyError("Colonne 'text' absente pour modality_mode='text' ou 'multimodal'.")
        if self.modality_mode in {"image", "multimodal"} and "path_to_image" not in self.df.columns:
            raise KeyError("Colonne 'path_to_image' absente pour modality_mode='image' ou 'multimodal'.")
        if "label" not in self.df.columns:
            raise KeyError("Colonne 'label' absente du DataFrame.")

        # Index propre
        self.df.reset_index(drop=True, inplace=True)

    def _load_tokenizer(self):
        """
        Charge le tokenizer/processor approprié selon le modèle.
        
        Pour BERT et modèles texte classiques : utilise load_model_and_tokenizer.
        Pour CLIP/BLIP : charge le CLIPProcessor/BlipProcessor directement,
        car load_model_and_tokenizer retourne None pour ces modèles.
        
        IMPORTANT: Ajuste aussi self.max_len selon le modèle :
          - CLIP : max 77 tokens (limite hard du position embedding)
          - BLIP : max 512 tokens
          - Autres : conserve le max_len passé en paramètre
        """
        model_name = self.config.model_name.lower()
        
        # CLIP : charger le tokenizer + forcer max_len=77
        if model_name == 'clip':
            from transformers import CLIPProcessor
            tokenizer = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32").tokenizer
            self.max_len = min(self.max_len, 77)  # CLIP hard limit
            print(f"✅ Tokenizer chargé pour CLIP (max_len={self.max_len})")
            return tokenizer
        # BLIP : charger le tokenizer + limiter à 512
        elif model_name == 'blip':
            from transformers import BlipProcessor
            tokenizer = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base").tokenizer
            self.max_len = min(self.max_len, 512)
            print(f"✅ Tokenizer chargé pour BLIP (max_len={self.max_len})")
            return tokenizer
        else:
            # BERT, Gemma, etc. : utiliser load_model_and_tokenizer
            from models.utils import load_model_and_tokenizer
            _, embedder_tokenizer, _, _ = load_model_and_tokenizer(self.config, n_concepts=1)
            if embedder_tokenizer is None:
                print(f"⚠️ Tokenizer None pour modèle '{model_name}' — la tokenisation dans __getitem__ échouera si modality_mode='text'")
            return embedder_tokenizer

    def __len__(self):
        return len(self.df)

    def _load_and_transform_image(self, path: str):
        img = _safe_image_open(path)
        if self.transform:
            return self.transform(img)
        return img

    def _get_text(self, t: str) -> str:
        t = str(t) if t is not None else ""
        # Padding/truncation simple pour un batch collate facile
        if len(t) > self.max_len:
            return t[:self.max_len]
        return t.ljust(self.max_len)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
    
        item: Dict[str, Any] = {
            "label": int(row["label"])
        }
    
        # Concepts
        if self.concept_cols:
            for concept_col in self.concept_cols:
                clean_name = concept_col
                item[clean_name] = torch.tensor(float(row[concept_col]), dtype=torch.float32)
            
            concepts = row[self.concept_cols].astype(float).values
            item["concepts"] = torch.tensor(concepts, dtype=torch.float32)
    
        # TEXTE
        if self.modality_mode in {"text", "multimodal"}:
            raw_text = str(row["text"]) if row["text"] is not None else ""
            
            if self.config.modality_mode == "text":
                # Mode text-only : tokenization complète
                encoded = self.embedder_tokenizer(
                    raw_text,
                    max_length=self.max_len,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                item["input_ids"] = encoded["input_ids"].squeeze(0)
                item["attention_mask"] = encoded["attention_mask"].squeeze(0)
                item["text"] = self._get_text(raw_text)
            else:
                # Mode multimodal : texte brut
                item["text"] = self._get_text(raw_text)
    
        # Image
        if self.modality_mode in {"image", "multimodal"}:
            img = self._load_and_transform_image(row["path_to_image"])
            item["image"] = img
            item["path_to_image"] = row["path_to_image"]
    
        return item


# ============================================================================
# PRÉPARATION DES DATALOADERS
# ============================================================================

def prepare_generic_dataloaders(
    config: Any,
    annotation: str,
    require_val: bool = False,
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader,
           pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
    """
    Prépare des DataLoaders génériques à partir des CSV:
      - {config.path_to_input_csv}/train_df_{annotation}.csv
      - {config.path_to_input_csv}/val_df_{annotation}.csv (optionnel)
      - {config.path_to_input_csv}/test_df_{annotation}.csv

    S'appuie sur load_from_csv() + encode_labels() définies plus haut.
    La base des colonnes de concepts est définie sur le TRAIN et réutilisée pour VAL/TEST.
    """
    # 1) Charger les splits avec encodage de labels cohérent
    df_train, df_val, df_test = load_from_csv(annotation=annotation, config=config, require_val=require_val)

    # 2) Définir le transform image (au besoin)
    img_transform = None
    if config.modality_mode in {"image", "multimodal"}:
        img_transform = transforms.Compose([
            ConvertToRGB(),
            transforms.Resize((getattr(config, "img_size", 224), getattr(config, "img_size", 224))),
            transforms.ToTensor()
        ])

    # 3) Déterminer select_most_frequent effectif (désactiver pour cb_llm)
    select_k = getattr(config, "select_most_frequent", None)
    if annotation.lower() == "cb_llm":
        if select_k is not None:
            print("⚠️  'cb_llm' détecté : sélection par fréquence désactivée (concepts continus).")
        select_k = None

    # 4) Dataset TRAIN = référence pour les concepts
    ds_train = GenericMultimodalDataset(
        df=df_train,
        config=config,
        annotation=annotation,
        modality_mode=config.modality_mode,
        max_len=getattr(config, "max_len", 512),
        transform=img_transform,
        select_most_frequent=select_k,
        class_list=getattr(config, "class_list", None),
        concept_cols_base=None,  # le train découvre et filtre les concepts
    )

    # 👉 Liste de concepts "gold" (même ordre pour tous les splits)
    concept_cols_base = ds_train.concept_cols

    # 5) Datasets VAL / TEST : on impose la même base de concepts
    ds_val = None
    if df_val is not None:
        ds_val = GenericMultimodalDataset(
            df=df_val,
            config=config,
            annotation=annotation,
            modality_mode=config.modality_mode,
            max_len=getattr(config, "max_len", 512),
            transform=img_transform,
            select_most_frequent=select_k,   # ignoré car concept_cols_base fourni
            class_list=getattr(config, "class_list", None),
            concept_cols_base=concept_cols_base,  # 👈 IMPORTANT
        )

    ds_test = GenericMultimodalDataset(
        df=df_test,
        config=config,
        annotation=annotation,
        modality_mode=config.modality_mode,
        max_len=getattr(config, "max_len", 512),
        transform=img_transform,
        select_most_frequent=select_k,       # ignoré car concept_cols_base fourni
        class_list=getattr(config, "class_list", None),
        concept_cols_base=concept_cols_base,  # 👈 IMPORTANT
    )

    # 6) DataLoaders
    train_loader = DataLoader(
        ds_train, batch_size=getattr(config, "batch_size", 32),
        shuffle=True, num_workers=getattr(config, "num_workers", 4), pin_memory=True
    )
    val_loader = None
    if ds_val is not None:
        val_loader = DataLoader(
            ds_val, batch_size=getattr(config, "batch_size", 32),
            shuffle=False, num_workers=getattr(config, "num_workers", 4), pin_memory=True
        )
    test_loader = DataLoader(
        ds_test, batch_size=getattr(config, "batch_size", 32),
        shuffle=False, num_workers=getattr(config, "num_workers", 4), pin_memory=True
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        df_train.reset_index(drop=True),
        (df_val.reset_index(drop=True) if df_val is not None else None),
        df_test.reset_index(drop=True)
    )