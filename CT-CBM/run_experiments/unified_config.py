import re
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import StepLR
import pandas as pd
import numpy as np
from tqdm import tqdm


# ============================================================================
# LIMITES DE TOKENISATION PAR MODÈLE
# ============================================================================
MODEL_MAX_TOKEN_LIMITS = {
    'clip': 77,     # Hard limit: position embeddings de CLIP
    'blip': 512,    # BLIP text encoder
    'gemma': 8192,  # Gemma 2B context window
}

class Config:
    """Configuration unifiée pour tous les modèles et datasets"""
    
    def __init__(self, 
                 model_name: str, 
                 dataset: str,  
                 annotation: str,
                 modality_mode : str,
                 combine_type : str,
                 path_to_input :str,
                 path_to_output: str
                ):
        # Paramètres communs
        self.annotation = annotation  # "C3M", "our_annotation"
        self.infra = "A100"  # "A100", "DATABRICKS"
        self.mode = 'joint'
        self.modality_mode = modality_mode #'image', 'text', 'multimodal'
        self.path_to_input = path_to_input
        self.path_to_output = path_to_output
        self.combine_type = combine_type
        
        # Paramètres spécifiques au modèle et dataset
        self.model_name = model_name
        self.DATASET = dataset
        
        # Configuration automatique basée sur le modèle
        self._configure_model()
        
        # Configuration automatique basée sur le dataset
        self._configure_dataset()
        
        # Paramètres communs à tous
        self._configure_common()
        
        # Configuration des chemins
        self._configure_paths()
        
        # Cas spéciaux détectés dans vos configs
        self._configure_special_cases()

        # important
        self.cb_llm_mode = False  # True if i am computing combined_score for CBLLM and False if not

        # ✅ Clamp max_len selon les limites du modèle (CLIP=77, etc.)
        self._clamp_max_len()

        
    def _configure_model(self):
        """Configure les paramètres spécifiques au modèle"""
        from scripts.clip_config import is_clip_family, get_clip_dim, CLIP_REGISTRY
        
        # Dimension basée sur le nom du modèle
        if self.model_name == 'gemma':
            self.dim = 2304
        elif is_clip_family(self.model_name):
            # Utilise le registre centralisé pour toutes les variantes CLIP/BLIP
            if self.modality_mode == 'image':
                self.dim = get_clip_dim(self.model_name, 'image')
            elif self.modality_mode == 'multimodal':
                if self.combine_type == 'concat':
                    self.dim = get_clip_dim(self.model_name, 'multimodal')
                elif self.combine_type == 'combine':
                    self.dim = get_clip_dim(self.model_name, 'text')  # single modality dim
                else:
                    print('enter correct value for combine_type for the modality_mode:', self.modality_mode)
            else:
                # text-only mode
                self.dim = get_clip_dim(self.model_name, 'text')
        else:
            import re
            self.dim = 1024 if re.search(r'large', self.model_name, re.IGNORECASE) else 768
        
        # Batch size spécifique par modèle
        model_batch_sizes = {
            'bert-base-uncased': 16,
            'deberta-large': 8,
            'gemma': 8,
            'clip': 32,
            'clip-large': 16,     # Plus gros modèle → batch plus petit
            'blip': 32,
        }
        self.batch_size = model_batch_sizes.get(self.model_name, 16)
        
        self.lambda_XtoC = 0.5

    
    def _configure_dataset(self):
        """Configure les paramètres spécifiques au dataset"""
        
        # Nombre de labels par dataset - exactement comme dans vos configs
        dataset_labels = {
            "dbpedia": 6,
            "agnews": 4,
            "movies": 4,
            "medical": 5,
            "ledgar": 4,
            "N24News": 4,
            "PBC":5,
            # "CUB": 200,
            "CUB": 15,
        }
        self.num_labels = dataset_labels.get(self.DATASET, 4)
        
        # Max length par dataset - basé sur vos commentaires et configs
        dataset_max_lengths = {
            "dbpedia": 128,    # "128 pour dbpedia et agnews"
            "agnews": 128,     # "128 pour dbpedia et agnews"
            "movies": 256,     # config_movies_deberta.py, config_movies_gemma.py
            "medical": 256,    # config_medical_gemma.py (pas 512)
            "ledgar": 256,     # config_ledgar_deberta.py
            "N24News": 512,     # config_n24news.py
            "CUB": 512     # config_n24news.py
        }
        self.max_len = dataset_max_lengths.get(self.DATASET, 256)
    
    def _configure_common(self):
        """Configure les paramètres communs à tous"""
        self.is_aux_logits = False
        self.num_epochs = 10
        self.num_each_concept_classes = 1  # Commentaire dans vos configs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.expand_dim = 0
        self.num_concept_labels = 4
        self.seed = 42
        self.n_concept_initial = 1
        self.criterion = nn.CrossEntropyLoss()
        self.use_cls_token = True  # Valeur par défaut for modality mode == 'text 
        self.use_relu = False
        self.sigmoid_or_relu_state  = 'linearity'
        self.agg_mode = "abs"
        self.agg_scope = "all"
        self.alpha = 0.01
        self.l1_ratio = 0.5
        self.l2_lambda = 0.01
        self.eval_concepts = True  # Par défaut, activer le calcul des métriques sur les concepts

    def _configure_paths(self):
        """Configure les chemins de stockage"""
        self.path_to_input_csv = f"{self.path_to_input}/{self.annotation}_annotation"
        
        # CUB utilise 'images' comme dossier, et N24 utilisE 'imgs'
        if self.DATASET == "CUB":
            self.path_to_input_images = f"{self.path_to_input}/../images"
        else:
            self.path_to_input_images = f"{self.path_to_input}/../imgs"

        self.SAVE_PATH = self.path_to_output
        
    # def _configure_special_cases(self):
    #     """Gère les cas spéciaux trouvés dans vos configs"""
    #     from scripts.clip_config import is_clip_family
    #     # Gestion des modèles multimodaux pour n24news
    #     if (self.DATASET == 'n24news') or (self.DATASET == 'CUB'):
    #     # Paramètres spécifiques pour les modèles multimodaux
    #         # self.combine_type = 'concat'# 'image', 'text', 'combine', 'concat'
    #         self.select_most_frequent = None 
    #         self.num_workers =  4
    #         self.use_cls_token= None # not used but keep for avoiding conflict
    #     elif is_clip_family(self.model_name):
    #     # Paramètres spécifiques pour les modèles d'image
    #         self.combine_type = 'image'
    #     else:
    #         pass

    def _configure_special_cases(self):
        from scripts.clip_config import is_clip_family, get_clip_dim
        
        if (self.DATASET == 'n24news') or (self.DATASET == 'CUB'):
            self.combine_type = 'concat'
            self.select_most_frequent = None 
            self.num_workers = 4
            self.use_cls_token = None
            # ✅ Recalculer dim avec le bon combine_type
            if is_clip_family(self.model_name) and self.modality_mode == 'multimodal':
                self.dim = get_clip_dim(self.model_name, 'multimodal')
        elif is_clip_family(self.model_name):
            self.combine_type = 'image'
        else:
            pass
 
        if self.DATASET == 'CUB':
            self.num_epochs = 15  # ✅ CUB a besoin de plus d'epochs (pas de val set, pas d'early stopping)


    # === REMPLACER _clamp_max_len() ===
    def _clamp_max_len(self):
        """Clamp max_len selon les limites architecturales du modèle."""
        from scripts.clip_config import is_clip_family, get_clip_max_tokens
        
        if is_clip_family(self.model_name):
            limit = get_clip_max_tokens(self.model_name)
            if self.max_len > limit:
                print(f"⚠️ max_len {self.max_len} → {limit} (limite {self.model_name})")
                self.max_len = limit
        elif self.model_name == 'gemma':
            limit = 8192
            if self.max_len > limit:
                self.max_len = limit
                
    def update_params(self, **kwargs):
        """Permet de modifier des paramètres après instantiation"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: {key} n'est pas un attribut valide de Config")
        
        # Re-clamp si max_len a été modifié manuellement
        if 'max_len' in kwargs:
            self._clamp_max_len()
    
    def __repr__(self):
        """Affichage lisible de la configuration"""
        return f"Config(model={self.model_name}, dataset={self.DATASET}, dim={self.dim}, max_len={self.max_len}, batch_size={self.batch_size})"


def load_config(model_name: str, dataset: str, annotation :str, modality_mode : str, combine_type : str, path_to_input :str, path_to_output: str):
    """Fonction simplifiée"""
    return Config(model_name, dataset, annotation, modality_mode, combine_type, path_to_input, path_to_output)