"""
Pipeline Dispatcher - Unified modality handling for CT-CBM
===========================================================

This module provides automatic modality detection and routing to appropriate
functions based on model type and data characteristics.

MODERNIZED VERSION: Uses cav_computation_unified.py for all CAV computations
and provides cleaner separation of concerns.

Author: CT-CBM Team
Date: 2025-11-06
Version: 2.0
"""

import torch
import os
import json
import pickle
import time
import gc
from typing import Tuple, Optional, Union, Dict, Any, List
import pandas as pd
from torch.utils.data import DataLoader

# Import du module unifié de calcul des CAVs
from cav_computation_unified import (
    compute_cavs_mean_minus_text,
    compute_cavs_mean_minus_gemma,
    compute_cavs_mean_minus_image,
    compute_cavs_mean_minus_multimodal,
)

from identifiability_score_R2_unified import (
    compute_r2_identifiability_score,
    compute_r2_identifiability_score_image,
    compute_r2_identifiability_score_multimodal
)

class ModalityDispatcher:
    """
    Central dispatcher that automatically detects modality and routes to appropriate functions.
    
    VERSION 2.0 FEATURES:
    - Uses cav_computation_unified.py for all CAV operations
    - Cleaner separation between modality types
    - Better error handling and validation
    - Modular method design for maintainability
    
    Attributes:
        config: Configuration object
        modality_mode: Detected or specified modality ("text", "multimodal", "image")
        is_gemma: Boolean flag for Gemma-specific handling
    
    Example:
        >>> config = load_config('clip', 'n24news')
        >>> dispatcher = ModalityDispatcher(config)
        >>> print(f"Detected mode: {dispatcher.modality_mode}")
        >>> data = dispatcher.load_data()
        >>> cavs = dispatcher.compute_cavs(data[3], baseline_model)
    """
    
    def __init__(self, config):
        """
        Initialize the dispatcher with configuration.
        
        Args:
            config: Configuration object with model_name, dataset, etc.
        """
        self.config = config
        self.modality_mode = self.config.modality_mode.lower() if hasattr(self.config, 'modality_mode') else 'text'
        self.is_gemma = True if 'gemma' in self.config.model_name.lower() else False
        self.is_image = True if self.config.modality_mode == 'image' else False
        self.is_multimodal = True if self.config.modality_mode == 'multimodal' else False
        # NEW: flag for CLIP/BLIP text-only mode (includes clip-large)
        from clip_config import is_clip_family
        self.is_clip_text = (
            is_clip_family(self.config.model_name)
            and not self.is_multimodal 
            and not self.is_image
        )
        
        print(f"🔍 ModalityDispatcher v2.0 initialized:")
        print(f"   - Mode: {self.modality_mode}")
        print(f"   - Model: {self.config.model_name}")
        print(f"   - Gemma: {self.is_gemma}")
        print(f"   - Image: {self.is_image}")
        print(f"   - Multimodal: {self.is_multimodal}")
        print(f"   - CLIP/BLIP Text-only: {self.is_clip_text}")
    
    
    def _is_cb_llm_annotation(self) -> bool:
        """
        Check if current annotation is CB-LLM.
        
        Returns:
            bool: True if annotation is 'cb_llm'
        """
        return hasattr(self.config, 'annotation') and self.config.annotation.lower() == 'cb_llm'
    
    def _validate_cb_llm_operation(self, operation_name: str) -> None:
        """
        Validate if an operation is supported for CB-LLM annotation.
        
        Args:
            operation_name: Name of the operation to validate
            
        Raises:
            ValueError: If operation is not supported for CB-LLM
        """
        if not self._is_cb_llm_annotation():
            return
        
        unsupported_operations = {
            'clustering': 'Clustering is not supported for CB-LLM annotation',
            'mean_minus_cavs': 'Mean-minus CAV computation is not supported for CB-LLM',
            'tcav_scores': 'TCAV scores computation may not work correctly with CB-LLM'
        }
        
        if operation_name in unsupported_operations:
            raise ValueError(
                f"❌ {unsupported_operations[operation_name]}\n"
                f"   CB-LLM annotation only supports:\n"
                f"   - R² identifiability score computation (compute_r2_score)\n"
                f"   - LIG ranking (compute_lig_ranking)\n"
                f"   For other operations, please use a different annotation (e.g., 'C3M', 'LLM')."
            )
    
    def _validate_data_source(self, data_source: Union[pd.DataFrame, DataLoader]) -> None:
        """
        Validate that data_source matches expected type for current modality.
        
        Args:
            data_source: DataFrame or DataLoader
            
        Raises:
            ValueError: If data_source type doesn't match modality requirements
        """
        if self.is_multimodal :
            if not isinstance(data_source, DataLoader):
                raise ValueError(
                    f"Multimodal models ({self.config.model_name}) require a DataLoader, "
                    f"got {type(data_source).__name__}"
                )
        elif self.is_image :
            if not isinstance(data_source, DataLoader):
                raise ValueError(
                    f"Image models ({self.config.model_name}) require a DataLoader, "
                    f"got {type(data_source).__name__}"
                )
        else:
            if not isinstance(data_source, pd.DataFrame):
                raise ValueError(
                    f"Text-only models ({self.config.model_name}) require a DataFrame, "
                    f"got {type(data_source).__name__}"
                )
    
    # ============================================================================
    # DATA LOADING
    # ============================================================================
    
    def load_data(self) -> Tuple:
        """
        Load data using appropriate loader based on dataset and modality.
        
        Special handling for 'cb_llm' annotation which uses pre-computed cosine similarities.
        
        Returns:
            Tuple of (train_loader, test_loader, val_loader, df_aug_train, df_aug_val, df_aug_test)
            or (train_loader, test_loader, val_loader, train_df, val_df, test_df)
        """
        from prepare_data import prepare_generic_dataloaders
        train_loader, test_loader, val_loader, df_aug_train, df_aug_val, df_aug_test = prepare_generic_dataloaders(
            config=self.config, annotation=self.config.annotation
        )
                    
        return train_loader, test_loader, val_loader, df_aug_train, df_aug_val, df_aug_test
    
    def _load_data_cb_llm(self) -> Tuple:
        """
        Load data specifically for CB-LLM annotation.
        CB-LLM uses pre-computed cosine similarities and doesn't support clustering.
        
        Returns:
            Tuple with CB-LLM specific data structure
        """
        print(f"📊 Loading CB-LLM annotation data...")
        
        # Load standard data loaders
        from prepare_data import load_fc_prepare_data
        prepare_data = load_fc_prepare_data(self.config.DATASET)
        train_loader, test_loader, val_loader, train_df, val_df, test_df = prepare_data(self.config)
        
        # Load CB-LLM specific data with pre-computed cosine similarities
        if not hasattr(self.config, 'SAVE_PATH_CONCEPTS'):
            raise ValueError(
                "CB-LLM annotation requires config.SAVE_PATH_CONCEPTS to be defined"
            )
        
        cb_llm_path = f"{self.config.SAVE_PATH_CONCEPTS}/df_with_topics_v4_CB_LLM_ACC.csv"
        
        if not os.path.exists(cb_llm_path):
            raise FileNotFoundError(
                f"CB-LLM data file not found: {cb_llm_path}\n"
                f"Please ensure the file exists or use a different annotation."
            )
        
        df_aug_train_cbllm = pd.read_csv(cb_llm_path)
        print(f"✓ Loaded CB-LLM accuracy data from {cb_llm_path}")
        print(f"   Shape: {df_aug_train_cbllm.shape}")
        
        # CB-LLM typically doesn't have a test set in the same format
        df_aug_test_cbllm = None
        
        return train_loader, test_loader, val_loader, df_aug_train_cbllm, val_df, df_aug_test_cbllm
    
    def extract_concept_list(self, df_aug_train: pd.DataFrame) -> List[str]:
        """
        Extract concept list from augmented DataFrame.
        
        Special handling for CB-LLM which uses column names starting with 'concept_'.
        
        Args:
            df_aug_train: Augmented training DataFrame with concept columns
        
        Returns:
            List of concept names
        """
        # Special handling for CB-LLM annotation
        concept_name_list = []
        
        # Choix du préfixe qui permet de désigner la colonne de concept
        prefix = 'concept_' 
        
        for column in df_aug_train.columns:
            if prefix not in column:
                continue
            concept_name = column.replace(prefix, '')
            # if sum !=0
            if(df_aug_train[column].sum() != 0):
                concept_name_list.append(concept_name)
        
        print(f"✓ Extracted {len(concept_name_list)} concepts")
        return concept_name_list

    # ============================================================================
    # BLACK-BOX MODEL
    # ============================================================================
    
    def load_or_train_blackbox(self,
                               train_loader: DataLoader,
                               val_loader: DataLoader,
                               test_loader: DataLoader,
                               force_retrain: bool = False) -> Tuple[Any, Optional[Any]]:
        """
        Load or train the black-box baseline model according to model type.
        
        This method handles the different BaselineModel classes for different
        architectures (Gemma, CLIP/BLIP, standard transformers).
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            force_retrain: If True, retrain even if checkpoint exists
        
        Returns:
            Tuple of (trained_model, tokenizer_or_processor)
        """
        print(f"🎯 Loading/training black-box model for {self.config.model_name}...")
        
        from models.utils import load_model_and_tokenizer
        
        # Determine the checkpoint path
        checkpoint_path = self._get_blackbox_checkpoint_path()
        
        # Load appropriate BaselineModel class and components
        if self.is_gemma:
            black_box_model, embedder_tokenizer = self._load_blackbox_gemma(
                train_loader, val_loader, test_loader, load_model_and_tokenizer
            )
        elif self.is_image:
            black_box_model, processor = self._load_blackbox_image(
                train_loader, val_loader, test_loader, load_model_and_tokenizer
            )
            embedder_tokenizer = processor  
        elif self.is_multimodal:
            black_box_model, processor = self._load_blackbox_multimodal(
                train_loader, val_loader, test_loader, load_model_and_tokenizer
            )
            embedder_tokenizer = processor
        # NEW: CLIP/BLIP text-only mode
        elif self.is_clip_text:
            black_box_model, processor = self._load_blackbox_clip_text(
                train_loader, val_loader, test_loader, load_model_and_tokenizer
            )
            embedder_tokenizer = processor
        else:
            # text bert type model
            black_box_model, embedder_tokenizer = self._load_blackbox_text(
                train_loader, val_loader, test_loader, load_model_and_tokenizer
            )
        
        # Load or train
        if not os.path.exists(checkpoint_path) or force_retrain:
            if force_retrain:
                print("🔄 Force retraining black-box model...")
            else:
                print("📚 No checkpoint found, training black-box model...")
            
            black_box_model.train_model()
            black_box_model.evaluate_model(test_loader, 'Test')
            
            if hasattr(black_box_model, 'save_performance_json'):
                black_box_model.save_performance_json()
            
            print(f"✓ Black-box model trained and saved to {checkpoint_path}")
        else:
            print(f"✓ Loading pre-trained black-box model from {checkpoint_path}")
            black_box_model.load_model()
        
        return black_box_model, embedder_tokenizer
    
    # def _get_blackbox_checkpoint_path(self) -> str:
    #     """Get the checkpoint path for black-box model."""
    #     if hasattr(self.config, 'sigmoid_or_relu_state') and self.config.sigmoid_or_relu_state != 'linearity':
    #         return (
    #             f"{self.config.SAVE_PATH}/blue_checkpoints/{self.config.model_name}/"
    #             f"BaselineModel/{self.config.model_name}_classifier_state_dict_{self.config.sigmoid_or_relu_state}.pth"
    #         )
    #     else:
    #         return (
    #             f"{self.config.SAVE_PATH}/blue_checkpoints/{self.config.model_name}/"
    #             f"BaselineModel/{self.config.model_name}_classifier_state_dict.pth"
    #         )

        
    def _get_blackbox_checkpoint_path(self) -> str:
        """Get the checkpoint path for black-box model.
        
        CLIP/BLIP text-only uses '_text_' suffix to distinguish from
        image-only and multimodal checkpoints of the same model.
        """
        base_dir = (
            f"{self.config.SAVE_PATH}/blue_checkpoints/{self.config.model_name}/BaselineModel"
        )
        
        # Suffix for sigmoid/relu variants
        state_suffix = ""
        if hasattr(self.config, 'sigmoid_or_relu_state') and self.config.sigmoid_or_relu_state != 'linearity':
            state_suffix = f"_{self.config.sigmoid_or_relu_state}"
        
        # CLIP/BLIP text-only: uses _text_ prefix to distinguish from image/multimodal
        if self.is_clip_text:
            return f"{base_dir}/{self.config.model_name}_text_classifier_state_dict{state_suffix}.pth"
        else:
            return f"{base_dir}/{self.config.model_name}_classifier_state_dict{state_suffix}.pth"
    
    
    def _load_blackbox_gemma(self, train_loader, val_loader, test_loader, load_fn):
        """Load Gemma black-box model."""
        from models.BaselineModel_gemma import BaselineModel
        
        embedder_model, embedder_tokenizer, _, classifier = load_fn(self.config, n_concepts=1)
        
        black_box_model = BaselineModel(
            embedder_model, classifier, 
            train_loader, val_loader, test_loader, 
            self.config, save_path=self.config.SAVE_PATH
        )
        
        return black_box_model, embedder_tokenizer
    
    def _load_blackbox_multimodal(self, train_loader, val_loader, test_loader, load_fn):
        """Load CLIP/BLIP black-box model."""
        from models.BaselineModel_clip import BaselineModel
        from models.BaselineModel_clip_CUB import BaselineModel as BaselineModel_CUB
        
        from clip_config import get_clip_checkpoint, is_clip_model
        from transformers import CLIPProcessor, BlipProcessor
        
        _, _, _, classifier = load_fn(self.config, n_concepts=1)
        
        # Load processor
        checkpoint = get_clip_checkpoint(self.config.model_name)
        if is_clip_model(self.config.model_name):
            processor = CLIPProcessor.from_pretrained(checkpoint)
        else:
            processor = BlipProcessor.from_pretrained(checkpoint)

        if self.config.DATASET == 'CUB':
            print("using personalised version of CUB baseline model blackbox")
            black_box_model = BaselineModel_CUB(
                None, classifier,
                train_loader, val_loader, test_loader,
                self.config, save_path=self.config.SAVE_PATH
            )
        else :
            black_box_model = BaselineModel(
                None, classifier,
                train_loader, val_loader, test_loader,
                self.config, save_path=self.config.SAVE_PATH
            )
        
        return black_box_model, processor

    def _load_blackbox_image(self, train_loader, val_loader, test_loader, load_fn):
        """Load CLIP/BLIP black-box model."""
        from models.BaselineModel_image import BaselineModel
        from clip_config import get_clip_checkpoint, is_clip_model
        from transformers import CLIPProcessor, BlipProcessor
        
        _, _, _, classifier = load_fn(self.config, n_concepts=1)
        
        # Load processor from centralized registry
        checkpoint = get_clip_checkpoint(self.config.model_name)
        if is_clip_model(self.config.model_name):
            processor = CLIPProcessor.from_pretrained(checkpoint)
        else:
            processor = BlipProcessor.from_pretrained(checkpoint)
        
        black_box_model = BaselineModel(
            None, classifier,
            train_loader, val_loader, test_loader,
            self.config, save_path=self.config.SAVE_PATH
        )
        
        return black_box_model, processor
    
    # def _load_blackbox_clip_text(self, train_loader, val_loader, test_loader, load_fn):
    #     """
    #     Load CLIP/BLIP text-only black-box model.
        
    #     Uses BaselineModel_clip_text which extracts only text embeddings
    #     from CLIP (get_text_features -> [B, 512]) or BLIP (text_model -> [B, 768]).
        
    #     Args:
    #         train_loader: Training data loader
    #         val_loader: Validation data loader  
    #         test_loader: Test data loader
    #         load_fn: Function to load model components
        
    #     Returns:
    #         Tuple of (BaselineModel, processor)
    #     """
    #     from models.BaselineModel_clip_text import BaselineModel
    #     from transformers import CLIPProcessor, BlipProcessor
        
    #     _, _, _, classifier = load_fn(self.config, n_concepts=1)
        
    #     # Load processor (used as tokenizer for text-only mode)
    #     if self.config.model_name == 'clip':
    #         processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    #     elif self.config.model_name == 'blip':
    #         processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    #     else:
    #         processor = None
        
    #     black_box_model = BaselineModel(
    #         None, classifier,
    #         train_loader, val_loader, test_loader,
    #         self.config, save_path=self.config.SAVE_PATH
    #     )
        
    #     return black_box_model, processor

    def _load_blackbox_clip_text(self, train_loader, val_loader, test_loader, load_fn):
        """
        Load CLIP/BLIP text-only black-box model.
        
        Uses BaselineModel_clip_text which extracts only text embeddings
        from CLIP (get_text_features -> [B, 512]) or BLIP (text_model -> [B, 768]).
        
        Now that load_model_and_tokenizer returns the proper tokenizer for CLIP/BLIP
        (CLIPTokenizerFast / BlipTokenizer), we use it directly instead of loading
        the full Processor. This ensures consistency across the entire pipeline.
        
        Returns:
            Tuple of (BaselineModel, tokenizer)
        """
        from models.BaselineModel_clip_text import BaselineModel
        
        # load_fn now returns proper tokenizer for CLIP/BLIP (not None)
        _, embedder_tokenizer, _, classifier = load_fn(self.config, n_concepts=1)
        
        black_box_model = BaselineModel(
            None, classifier,
            train_loader, val_loader, test_loader,
            self.config, save_path=self.config.SAVE_PATH
        )
        
        return black_box_model, embedder_tokenizer
    
    def _load_blackbox_text(self, train_loader, val_loader, test_loader, load_fn):
        """Load standard text black-box model."""
        from models.BaselineModel_text import BaselineModel
        
        embedder_model, embedder_tokenizer, _, classifier = load_fn(self.config, n_concepts=1)
        
        black_box_model = BaselineModel(
            embedder_model, classifier,
            train_loader, val_loader, test_loader,
            self.config, save_path=self.config.SAVE_PATH
        )
        
        return black_box_model, embedder_tokenizer
    # ============================================================================
    # CAV COMPUTATION (Using cav_computation_unified.py)
    # ============================================================================
    
    def compute_cavs(self, 
                     data_source: Union[pd.DataFrame, DataLoader], 
                     baseline_model,
                     concept_list: Optional[List[str]] = None,
                     use_images: bool = True,
                     embedder_tokenizer=None,
                     processor=None) -> Dict[str, torch.Tensor]:
        """
        Compute CAVs using the unified cav_computation module.
        
        IMPORTANT: CB-LLM annotation does NOT support mean-minus CAV computation.
        For CB-LLM, use compute_r2_score() instead to get CAVs from regression.
        
        Args:
            data_source: DataFrame (for text) or DataLoader (for multimodal/image)
            baseline_model: The baseline model
            concept_list: Optional list of concepts to compute CAVs for
            use_images: Whether to use images (for multimodal mode)
            embedder_tokenizer: Tokenizer (needed for text-only modes)
            processor: Processor (needed for multimodal modes)
        
        Returns:
            Dict mapping concept names to CAV vectors (numpy arrays)
            
        Raises:
            ValueError: If called with CB-LLM annotation (use compute_r2_score instead)
        """
        print(f"🧮 Computing CAVs in {self.modality_mode} mode...")
        
        # CB-LLM annotation check
        if self._is_cb_llm_annotation():
            print(
                "❌ CB-LLM annotation does not support mean-minus CAV computation.\n"
                "   CB-LLM CAVs can ONLY be computed via R² identifiability score.\n"
                "   dispatcher.compute_r2_score() will be used instead.\n"
                "   Example:\n"
                "   >>> metrics, filtered, cavs = dispatcher.compute_r2_score(\n"
                "   ...     data_source=df_with_cosines,\n"
                "   ...     embedder_model=model,\n"
                "   ...     embedder_tokenizer=tokenizer\n"
                "   ... )"
            )
            # - Text/Gemma: (train_df, val_df, metrics, filtered_concepts, cavs)
            # - Image/Multimodal: (metrics, filtered_concepts, cavs)
            
            result_here = self.compute_r2_score(
                data_source=data_source,
                embedder_model=baseline_model,      # cohérent
                embedder_tokenizer=embedder_tokenizer,
                processor=processor
            )
            
            # Renvoie les CAVs (toujours en dernière position dans les deux modes)
            return result_here[-1]
            
        # Validation
        self._validate_data_source(data_source)
        
        # Route to appropriate function
        if self.is_gemma:
            return self._compute_cavs_gemma(data_source, baseline_model, embedder_tokenizer)
        elif self.is_multimodal:
            return self._compute_cavs_multimodal(
                data_source, baseline_model, processor, concept_list, use_images
            )
        elif self.is_image:
            return self._compute_cavs_image(
                data_source, baseline_model, concept_list
            )
        else:
            # Text-only (BERT or CLIP/BLIP text-only) - same CAV computation logic
            return self._compute_cavs_text(data_source, baseline_model, embedder_tokenizer)
    
    def _compute_cavs_text(self, 
                          df_aug: pd.DataFrame,
                          baseline_model,
                          embedder_tokenizer) -> Dict[str, torch.Tensor]:
        """Compute CAVs for standard text models (BERT, RoBERTa, etc.) and CLIP/BLIP text-only."""
        if embedder_tokenizer is None:
            raise ValueError("embedder_tokenizer is required for text-only models")
        
        cavs = compute_cavs_mean_minus_text(
            df_aug=df_aug,
            baseline_model=baseline_model,
            tokenizer=embedder_tokenizer,
            config=self.config
        )
        
        print(f"✓ Computed {len(cavs)} CAVs (text mode)")
        return cavs
    
    def _compute_cavs_gemma(self,
                           df_aug: pd.DataFrame,
                           baseline_model,
                           embedder_tokenizer) -> Dict[str, torch.Tensor]:
        """Compute CAVs for Gemma models."""
        if embedder_tokenizer is None:
            raise ValueError("embedder_tokenizer is required for Gemma models")
        
        cavs = compute_cavs_mean_minus_gemma(
            df_aug=df_aug,
            baseline_model=baseline_model,
            tokenizer=embedder_tokenizer,
            config=self.config
        )
        
        print(f"✓ Computed {len(cavs)} CAVs (gemma mode)")
        return cavs
    
    def _compute_cavs_multimodal(self,
                                loader: DataLoader,
                                baseline_model,
                                processor,
                                concept_list: Optional[List[str]],
                                use_images: bool) -> Dict[str, torch.Tensor]:
        """Compute CAVs for multimodal models (CLIP/BLIP)."""
        if processor is None:
            raise ValueError("processor is required for multimodal models")
        
        cavs = compute_cavs_mean_minus_multimodal(
            loader=loader,
            baseline_model=baseline_model,
            processor=processor,
            config=self.config,
            concept_list=concept_list,
            use_images=use_images
        )
        
        print(f"✓ Computed {len(cavs)} CAVs (multimodal mode)")
        return cavs
    
    def _compute_cavs_image(self,
                           loader: DataLoader,
                           baseline_model,
                           concept_list: Optional[List[str]]) -> Dict[str, torch.Tensor]:
        """Compute CAVs for image-only models."""
        cavs = compute_cavs_mean_minus_image(
            loader=loader,
            baseline_model=baseline_model,
            config=self.config,
            concept_list=concept_list
        )
        
        print(f"✓ Computed {len(cavs)} CAVs (image mode)")
        return cavs
 
    # ============================================================================
    # CAV MANAGEMENT (Load/Save)
    # ============================================================================
    
    def load_or_compute_cavs(self,
                            data_source: Union[pd.DataFrame, DataLoader],
                            baseline_model,
                            concept_list: Optional[List[str]] = None,
                            embedder_tokenizer=None,
                            processor=None,
                            force_recompute: bool = False,
                            use_images: bool = True) -> Dict[str, torch.Tensor]:
        """
        Load CAVs from file or compute them if they don't exist.
        
        Args:
            data_source: DataFrame or DataLoader depending on modality
            baseline_model: The baseline model
            concept_list: List of concept names
            embedder_tokenizer: Tokenizer for text models
            processor: Processor for multimodal models
            force_recompute: If True, recompute even if file exists
            use_images: Whether to use images (multimodal only)
        
        Returns:
            Dictionary of CAVs
        """
        PATH_TO_CAVS = self._get_cavs_path()
        
        if not os.path.exists(PATH_TO_CAVS) or force_recompute:
            if force_recompute:
                print("🔄 Force recomputing CAVs...")
            else:
                print("📚 No CAVs found, computing...")
            
            cavs = self.compute_cavs(
                data_source=data_source,
                baseline_model=baseline_model,
                concept_list=concept_list,
                embedder_tokenizer=embedder_tokenizer,
                processor=processor,
                use_images=use_images
            )
            
            print(f"✓ CAVs computed and saved to {PATH_TO_CAVS}")
        else:
            print(f"✓ Loading CAVs from {PATH_TO_CAVS}")
            with open(PATH_TO_CAVS, 'r') as f:
                cavs_dict = json.load(f)
            # Convert to numpy arrays
            import numpy as np
            cavs = {k: np.array(v) for k, v in cavs_dict.items()}
        
        return cavs
    
    def _get_cavs_path(self) -> str:
        """Get the path to CAVs file."""
        
        return (
            f"{self.config.SAVE_PATH}/blue_checkpoints/{self.config.model_name}/"
            f"cavs/{self.config.cavs_type}/cavs_{self.config.cavs_type}_{self.config.annotation}.json"
        )
 

    
    # ============================================================================
    # TCAV RANKER
    # ============================================================================
    
    def create_tcav(self, 
                    concepts: List[str], 
                    baseline_model,
                    embedder_tokenizer: Optional[Any] = None,
                    batch_size: Optional[int] = None,
                    verbose: bool = True):
        """
        Create TCAV instance appropriate for current modality.
        
        WARNING: TCAV may not work correctly with CB-LLM annotation.
        CB-LLM is designed for R² identifiability and LIG ranking.
        
        Args:
            concepts: List of concept names
            baseline_model: The baseline model
            embedder_tokenizer: Tokenizer (needed for text modes)
            batch_size: Batch size (defaults to config.batch_size)
            verbose: Whether to print verbose output
        
        Returns:
            TCAV instance configured for current modality
        """
        print(f"🎯 Creating TCAV ranker in {self.modality_mode} mode...")
        
        # Warning for CB-LLM
        if self._is_cb_llm_annotation():
            print("⚠️  WARNING: CB-LLM annotation may not work correctly with TCAV scores.")
            print("   CB-LLM is optimized for R² identifiability and LIG ranking.")
            print("   Consider using a different annotation for TCAV-based ranking.")
        
        batch_size = batch_size or self.config.batch_size
        
        # Import the appropriate TCAV class
        if self.is_multimodal:
            from TCAVS_multimodal import TCAV
        elif self.is_image:
            from TCAVS_image import TCAV
        elif self.is_gemma:
            from TCAVS_gemma import TCAV
        else:
            # Text-only: works for both BERT and CLIP/BLIP text-only
            from TCAVS_text import TCAV
        
        tcav_ranker = TCAV(
            concepts=concepts,
            baseline_model=baseline_model,
            embedder_tokenizer=embedder_tokenizer,
            batch_size=batch_size,
            config=self.config,
            verbose=verbose
        )
        
        return tcav_ranker
    
    def compute_tcav_scores(self,
                           tcav_ranker,
                           train_loader: DataLoader,
                           max_examples_per_concept: int = 2) -> Dict:
        """
        Compute TCAV scores using the ranker.
        
        Args:
            tcav_ranker: TCAV instance
            train_loader: Training data loader
            max_examples_per_concept: Maximum examples per concept
        
        Returns:
            Dictionary of TCAV scores by class
        """
        print(f"📊 Computing TCAV scores...")
        
        # Load CAVs from file
        tcav_ranker.load_cavs_from_file()
        
        # Calculate TCAV scores
        tcav_scores = tcav_ranker.calculate_tcav_scores_non_linearity_hypothesis_20(
            train_loader,
            max_examples_per_concept=max_examples_per_concept
        )
        
        print(f"✓ TCAV scores computed")
        return tcav_scores
    
    def rank_and_save_concepts(self, tcav_scores: Dict) -> List:
        """
        Rank concepts based on TCAV scores and save to file.
        
        Args:
            tcav_scores: Dictionary of TCAV scores
        
        Returns:
            List of sorted macro concepts
        """
        from ranking_utils import rank_macro_concepts
        
        # Rank the macro concepts based on TCAV scores
        sorted_macro_concepts = rank_macro_concepts(tcav_scores)
        
        # Save to file
        file_path = (
            f"{self.config.SAVE_PATH}/blue_checkpoints/{self.config.model_name}/"
            f"cavs/{self.config.cavs_type}"
        )
        os.makedirs(file_path, exist_ok=True)
        
        output_path = f'{file_path}/TCAVS_scores_{self.config.annotation}.json'
        with open(output_path, 'w') as f:
            json.dump(sorted_macro_concepts, fp=f, default=lambda x: x.tolist())
        
        print(f'✓ TCAVS score saved to: {output_path}')
        return sorted_macro_concepts
    
    
    # ============================================================================
    # LIG (Layer Integrated Gradients) RANKING
    # ============================================================================

        
    
    def _prepare_cavs_tensors(self, cavs):
        """Convert CAVs to tensors on correct device."""
        if isinstance(cavs, str):
            with open(cavs, 'r') as f:
                cavs_vectors = json.load(f)
            return {k: torch.tensor(v, dtype=torch.float32).to(self.config.device) 
                   for k, v in cavs_vectors.items()}
        elif isinstance(cavs, dict):
            return {k: (v.to(self.config.device) if isinstance(v, torch.Tensor) 
                       else torch.tensor(v, dtype=torch.float32).to(self.config.device))
                   for k, v in cavs.items()}
        else:
            raise ValueError(f"Unsupported cavs type: {type(cavs)}")
    
    def _compute_or_load_cosine_similarities(self, attributions_df, train_df, cavs, force_recompute):
        """Compute or load cosine similarities."""
        path_cosine_df = (
            f"{self.config.SAVE_PATH}/blue_checkpoints/{self.config.model_name}/"
            f"cavs/{self.config.cavs_type}/df_aug_train_updated_{self.config.annotation}.pkl"
        )
        
        if os.path.exists(path_cosine_df) and not force_recompute:
            print("📂 Loading existing cosine similarities...")
            with open(path_cosine_df, "rb") as f:
                return pickle.load(f)
        
        print("🧮 Computing cosine similarities...")
        
        # Import appropriate function
        if self.is_gemma:
            from LIG_ranking_text import compute_cosine_similarities
        elif self.is_image :
            from LIG_ranking_image import compute_cosine_similarities
        elif self.is_multimodal :
            from LIG_ranking_multimodal import compute_cosine_similarities
        else:
            # Text-only: works for both BERT and CLIP/BLIP text-only
            from LIG_ranking_text import compute_cosine_similarities
        
        df_aug_train_updated = compute_cosine_similarities(
            attributions_df, train_df, cavs, self.config.device
        )
        
        with open(path_cosine_df, "wb") as f:
            pickle.dump(df_aug_train_updated, f)
        print(f"✓ Cosine similarities saved to {path_cosine_df}")
        
        return df_aug_train_updated
    
    def _compute_or_load_lig_scores(self, df_aug, cavs, mode, agg_scope, 
                                        force_recompute, postprocess_fn):
        """Compute or load sorted concepts."""
        file_path_sorted = (
            f"{self.config.SAVE_PATH}/blue_checkpoints/{self.config.model_name}/"
            f"cavs/{self.config.cavs_type}/LIG_scores_"
            f"{self.config.cavs_type}_{self.config.annotation}_{mode}_{agg_scope}.json"
        )
        
        if os.path.exists(file_path_sorted) and not force_recompute:
            print("📂 Loading existing sorted concepts...")
            with open(file_path_sorted, "r") as f:
                return json.load(f)
        
        print("🧮 Sorting concepts by LIG scores...")
        df_aug_updated, sorted_concepts = postprocess_fn(
            df_aug, list(cavs.keys()), mode=mode, agg_scope=agg_scope
        )
        
        with open(file_path_sorted, "w") as f:
            json.dump(sorted_concepts, f, indent=4)
        print(f"✓ Sorted concepts saved to {file_path_sorted}")
        
        return sorted_concepts

    # ============================================================================
    # LIG (Layer Integrated Gradients) RANKING
    # ============================================================================
    
    def create_lig_wrapper(self, black_box_model):
        """
        Create LayerIntegratedGradients wrapper for the black-box model.
        
        Args:
            black_box_model: The trained black-box model
        
        Returns:
            Tuple of (forward_function, lig_instance)
        """
        from captum.attr import LayerIntegratedGradients
        
        print(f"🔬 Creating LIG wrapper for {self.config.model_name}...")
        
        if self.is_gemma:
            forward_fn, lig = self._create_lig_gemma(black_box_model, LayerIntegratedGradients)
        elif self.is_image:
            forward_fn, lig = self._create_lig_image(black_box_model, LayerIntegratedGradients)
        elif self.is_multimodal:
            forward_fn, lig = self._create_lig_multimodal(black_box_model, LayerIntegratedGradients)
        elif self.is_clip_text:
            # CLIP/BLIP text-only: uses concat_layer like image/multimodal
            forward_fn, lig = self._create_lig_clip_text(black_box_model, LayerIntegratedGradients)
        else:
            forward_fn, lig = self._create_lig_text(black_box_model, LayerIntegratedGradients)
        
        print(f"✓ LIG wrapper created")
        return forward_fn, lig
    
    def _create_lig_gemma(self, model, LIG):
        """Create LIG wrapper for Gemma."""
        def forward_LIG_black_box(input_ids, attention_mask=None):
            outputs = model.embedder_model(input_ids=input_ids, attention_mask=attention_mask)
            pooled = outputs[0][:, -1, :]
            logits = model.classifier(pooled)
            return logits
        
        lig = LIG(forward_LIG_black_box, layer=model.embedder_model.layers[-1])
        return forward_LIG_black_box, lig
    
    def _create_lig_multimodal(self, model, LIG):
        """Create LIG wrapper for CLIP/BLIP multimodal."""
        def forward_LIG_black_box(pooled):
            pooled = model.concat_layer(pooled)
            logits = model.classifier(pooled)
            return logits
        
        lig = LIG(forward_LIG_black_box, layer=model.concat_layer)
        return forward_LIG_black_box, lig

    def _create_lig_image(self, model, LIG):
        """Create LIG wrapper for CLIP/BLIP image-only."""
        def forward_LIG_black_box(pooled):
            pooled = model.concat_layer(pooled)
            logits = model.classifier(pooled)
            return logits
        
        lig = LIG(forward_LIG_black_box, layer=model.concat_layer)
        return forward_LIG_black_box, lig
    
    
    # def _create_lig_clip_text(self, model, LIG):
    #     """
    #     Create LIG wrapper for CLIP/BLIP text-only.
        
    #     Takes (input_ids, attention_mask) as inputs — same interface as _create_lig_text.
    #     Performs full CLIP/BLIP text encoding, then passes through classifier.
        
    #     The LIG layer target is the last layer of the text encoder, so attributions
    #     are computed on the text token representations.
    #     """
    #     device = self.config.device
    #     is_clip = getattr(model, '_is_clip', True)
        
    #     def forward_LIG_black_box(input_ids, attention_mask):
    #         input_ids = input_ids.to(device)
    #         attention_mask = attention_mask.to(device)
            
    #         # Get text embeddings via the model's own method
    #         pooled_output = model.get_pooled_output(input_ids, attention_mask)
    #         pooled_output = model.concat_layer(pooled_output)
    #         logits = model.classifier(pooled_output)
    #         return logits
        
    #     # Select the appropriate text encoder layer for LIG attribution
    #     if is_clip:
    #         # CLIP: text_model.encoder.layers[-1]
    #         target_layer = model.embedder_model.text_model.encoder.layers[-1]
    #     else:
    #         # BLIP: text_model.encoder.layer[-1]
    #         target_layer = model.embedder_model.text_model.encoder.layer[-1]
        
    #     lig = LIG(forward_LIG_black_box, layer=target_layer)
    #     return forward_LIG_black_box, lig
    
    # For CLIP/BLIP text-only, we target the concat_layer (nn.Identity after get_pooled_output) 
    def _create_lig_clip_text(self, model, LIG):
        """
        Create LIG wrapper for CLIP/BLIP text-only.
        
        Takes (input_ids, attention_mask) as inputs — same interface as _create_lig_text.
        
        CRITICAL: Target layer = concat_layer (nn.Identity after get_pooled_output).
        This ensures attributions are in the SAME projected space as the CAVs.
        Using text_model.encoder.layers[-1] gives attributions in hidden space
        (before text_projection) → different vector space → cosine with CAVs ≈ 0.
        
        Attribution shape per sample: (projected_dim,) i.e. (512,) for CLIP base.
        This is 1D, NOT (seq_len, dim) like BERT. compute_similarity_cav_text
        must handle both cases.
        """
        device = self.config.device
        
        def forward_LIG_black_box(input_ids, attention_mask):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            # Full forward: text encoder + text_projection + classifier
            pooled_output = model.get_pooled_output(input_ids, attention_mask)
            pooled_output = model.concat_layer(pooled_output)
            logits = model.classifier(pooled_output)
            return logits
        
        # Target = concat_layer → attributions in projected space = same as CAVs
        lig = LIG(forward_LIG_black_box, layer=model.concat_layer)
        return forward_LIG_black_box, lig
        
    # " FOR BERT BACKBONES AND SIMILAR MODELS (ROBERTA, DISTILBERT, ETC.) "
    def _create_lig_text(self, model, LIG):
        """Create LIG wrapper for standard transformers (BERT, etc.)."""
        def forward_LIG_black_box(input_ids, attention_mask):
            outputs = model.embedder_model(
                input_ids=input_ids.to(self.config.device),
                attention_mask=attention_mask.to(self.config.device)
            )
            
            pooled_output = outputs.last_hidden_state[:, 0, :]

            logits = model.classifier(pooled_output)
            return logits
        
        lig = LIG(forward_LIG_black_box, layer=model.embedder_model.encoder.layer[-1])
        return forward_LIG_black_box, lig


        
    def compute_lig_ranking(self,
                           black_box_model,
                           train_loader: DataLoader,
                           train_df: pd.DataFrame,
                           cavs: Dict,
                           mode: str = "abs",
                           agg_scope: str = "all",
                           batch_size: int = 4,
                           force_recompute: bool = False):
        """
        Compute LIG (Layer Integrated Gradients) ranking for concepts.
        
        This is the COMPLETE LIG ranking pipeline that:
        1. Computes attributions
        2. Computes cosine similarities with CAVs
        3. Sorts concepts by aggregated scores
        
        Args:
            black_box_model: Trained black-box model
            train_loader: Training data loader
            train_df: Training DataFrame
            cavs: Dictionary of CAV vectors
            mode: Aggregation mode ("abs", "clip", etc.)
            agg_scope: Aggregation scope ("all", "present", etc.)
            batch_size: Batch size for processing
            force_recompute: If True, recompute even if files exist
        
        Returns:
            Tuple of (df_aug_train_updated, sorted_concepts)
        """
        print(f"🔬 Computing LIG ranking (mode={mode}, scope={agg_scope})...")
        start_time = time.time()
        
        # Create LIG wrapper
        forward_fn, lig = self.create_lig_wrapper(black_box_model)


        # Import appropriate functions based on model type
        if self.is_gemma:
            from LIG_ranking_text import compute_attributions_on_gemma as compute_attributions
            from LIG_ranking_text import postprocess_cosine, compute_cosine_similarities
        elif self.is_image:
            from LIG_ranking_image import compute_attributions_from_dataloader
            from LIG_ranking_image import postprocess_cosine, compute_cosine_similarities
        elif self.is_multimodal:
            from LIG_ranking_multimodal import compute_attributions_from_dataloader
            from LIG_ranking_multimodal import postprocess_cosine, compute_cosine_similarities
        else:
            # Text-only (BERT and CLIP/BLIP text-only)
            from LIG_ranking_text import compute_attributions
            from LIG_ranking_text import postprocess_cosine, compute_cosine_similarities
        
        # Compute attributions
        attributions_df = self._compute_or_load_attributions(
            train_loader,train_df , black_box_model, lig, force_recompute
        )
        
        # Load/convert CAVs to tensors
        cavs_tensors = self._prepare_cavs_tensors(cavs)
        
        # Compute cosine similarities
        df_aug_train_updated = self._compute_or_load_cosine_similarities(
            attributions_df, train_df, cavs_tensors, force_recompute
        )
        
        # Sort concepts
        sorted_concepts = self._compute_or_load_lig_scores(
            df_aug_train_updated, cavs_tensors, mode, agg_scope, 
            force_recompute, postprocess_cosine
        )
        
        # Cleanup
        del attributions_df
        gc.collect()
        
        elapsed_time = time.time() - start_time
        print(f"✓ LIG ranking complete in {elapsed_time:.2f} seconds")
        
        return df_aug_train_updated, sorted_concepts
    
    def compute_lig_scores(self,
                          black_box_model,
                          train_loader: DataLoader,
                          train_df: pd.DataFrame,
                          cavs: Dict,
                          force_recompute: bool = False) -> pd.DataFrame:
        """
        Compute LIG scores (attributions + cosine similarities) WITHOUT ranking.
        
        This is a SIMPLIFIED version that only computes the scores, not the final ranking.
        Use this if you want to do custom aggregation/ranking later.
        
        Args:
            black_box_model: Trained black-box model
            train_loader: Training data loader
            train_df: Training DataFrame
            cavs: Dictionary of CAV vectors
            force_recompute: If True, recompute even if files exist
        
        Returns:
            DataFrame with LIG scores (cosine similarities) for each concept
        """
        print(f"🔬 Computing LIG scores (no ranking)...")
        start_time = time.time()
        
        # Create LIG wrapper
        forward_fn, lig = self.create_lig_wrapper(black_box_model)
        
        # Compute attributions
        attributions_df = self._compute_or_load_attributions(
            train_loader, train_df, black_box_model, lig, force_recompute
        )
        
        # Load/convert CAVs to tensors
        cavs_tensors = self._prepare_cavs_tensors(cavs)
        
        # Compute cosine similarities
        df_with_scores = self._compute_or_load_cosine_similarities(
            attributions_df, train_df, cavs_tensors, force_recompute
        )
        
        # Cleanup
        del attributions_df
        gc.collect()
        
        elapsed_time = time.time() - start_time
        print(f"✓ LIG scores computed in {elapsed_time:.2f} seconds")
        
        return df_with_scores
        
    
    def _compute_or_load_attributions(self, train_loader, train_df, black_box_model, lig, force_recompute):
        """Compute or load attributions."""
        path_attr = (
            f"{self.config.SAVE_PATH}/blue_checkpoints/{self.config.model_name}/"
            f"cavs/{self.config.cavs_type}/attributions_df_{self.config.cavs_type}_{self.config.annotation}.pkl"
        )
        os.makedirs(os.path.dirname(path_attr), exist_ok=True)
        
        if os.path.exists(path_attr) and not force_recompute:
            print("📂 Loading existing attributions...")
            with open(path_attr, "rb") as f:
                return pickle.load(f)
        
        print("🧮 Computing attributions...")

      # need to obtain text tokenizer which are not saved in blackbox_model text here
        from models.utils import load_model_and_tokenizer
        _, embedder_tokenizer, _, _ = load_model_and_tokenizer(self.config, n_concepts=1)
        
        # Fallback for CLIP/BLIP: load_model_and_tokenizer returns None for these models
        if embedder_tokenizer is None:
            from clip_config import is_clip_family, get_clip_checkpoint, is_clip_model
            if is_clip_family(self.config.model_name):
                checkpoint = get_clip_checkpoint(self.config.model_name)
                if is_clip_model(self.config.model_name):
                    from transformers import CLIPProcessor
                    embedder_tokenizer = CLIPProcessor.from_pretrained(checkpoint).tokenizer
                else:
                    from transformers import BlipProcessor
                    embedder_tokenizer = BlipProcessor.from_pretrained(checkpoint).tokenizer
                print(f"✅ Tokenizer fallback: {checkpoint} tokenizer chargé")

        if self.config.modality_mode == 'image':
            from LIG_ranking_image import compute_attributions_from_dataloader
            attributions_df = compute_attributions_from_dataloader(
                train_loader, black_box_model,
                lig, self.config.device
            )
        elif self.is_multimodal:
            from LIG_ranking_multimodal import compute_attributions_from_dataloader
            attributions_df = compute_attributions_from_dataloader(
                train_loader, black_box_model,
                black_box_model.tokenizer if hasattr(black_box_model, 'tokenizer') else None,
                lig, self.config.device
            )
        elif self.is_gemma:
            from LIG_ranking_text import compute_attributions_on_gemma
            attributions_df = compute_attributions_on_gemma(
                train_df, self.config.batch_size, embedder_tokenizer, lig, self.config.device
            )
        elif self.is_clip_text:
            # CLIP/BLIP text-only: compute attributions from pooled text embeddings
            # Uses same pattern as image (attributions on pooled output via concat_layer)
            from LIG_ranking_text import compute_attributions
            attributions_df = compute_attributions(
                train_df, self.config.batch_size, embedder_tokenizer, lig, self.config.device
            )
        else:
            from LIG_ranking_text import compute_attributions
            attributions_df = compute_attributions(
                train_df, self.config.batch_size, embedder_tokenizer, lig, self.config.device
            )
        
        with open(path_attr, "wb") as f:
            pickle.dump(attributions_df, f)
        print(f"✓ Attributions saved to {path_attr}")
        
        return attributions_df
    
    

   
    # ============================================================================
    # R² IDENTIFIABILITY SCORE
    # ============================================================================
    
    def compute_r2_score(self,
                        data_source: Union[pd.DataFrame, DataLoader],
                        embedder_model,
                        embedder_tokenizer=None,
                        processor=None,
                        concept_columns: Optional[List[str]] = None,
                        r2_cutoff: Optional[float] = None,
                        test_size: float = 0.3,
                        random_state: int = 42,
                        save_cavs: bool = True,
                        use_images: bool = True):
        """
        Compute R² identifiability score using the unified module.
        
        This is the PRIMARY and ONLY method to compute CAVs for CB-LLM annotation.
        
        Args:
            data_source: DataFrame with cosine similarities (text) or 
                        DataLoader with 'cos_*' keys (multimodal/image)
            embedder_model: Model for extracting embeddings
            embedder_tokenizer: Tokenizer (for text modes)
            processor: Processor (for multimodal modes)
            concept_columns: List of concept names (auto-detected if None)
            r2_cutoff: Threshold for filtering concepts
            test_size: Validation set proportion
            random_state: Random seed
            save_cavs: Whether to save computed CAVs
            use_images: Whether to use images (multimodal only)
        
        Returns:
            Tuple depending on modality:
            - Text/Gemma: (train_df, val_df, metrics, filtered_concepts, cavs)
            - Image/Multimodal: (metrics, filtered_concepts, cavs)
        """
        print(f"📊 Computing R² identifiability score in {self.modality_mode} mode...")
        
        # Special message for CB-LLM
        if self._is_cb_llm_annotation():
            print("🔷 CB-LLM annotation detected: Using R² regression method for CAV computation")
            print("   This is the only supported method for CB-LLM CAVs.")
        
        if self.is_multimodal:
            return self._compute_r2_multimodal(
                data_source, embedder_model, processor, concept_columns,
                r2_cutoff, test_size, random_state, save_cavs, use_images
            )
        elif self.modality_mode == 'image':
            return self._compute_r2_image(
                data_source, embedder_model, concept_columns,
                r2_cutoff, test_size, random_state, save_cavs
            )
        else:
            # Text-only (BERT, Gemma, or CLIP/BLIP text-only)
            return self._compute_r2_text(
                data_source, embedder_model, embedder_tokenizer, concept_columns,
                r2_cutoff, test_size, random_state, save_cavs
            )
    
    def _compute_r2_text(self, df_cosine, embedder_model, embedder_tokenizer,
                        concept_columns, r2_cutoff, test_size, random_state, save_cavs):
        """Compute R² score for text/gemma models."""
        if embedder_tokenizer is None:
            raise ValueError("embedder_tokenizer is required for text models")
        
        model_type = 'gemma' if self.is_gemma else 'bert'
        
        return compute_r2_identifiability_score(
            df_cosine=df_cosine,
            embedder_model=embedder_model,
            embedder_tokenizer=embedder_tokenizer,
            config=self.config,
            concept_columns=concept_columns,
            r2_cutoff=r2_cutoff,
            test_size=test_size,
            random_state=random_state,
            save_cavs=save_cavs
        )
    
    def _compute_r2_image(self, loader, embedder_model, concept_columns,
                         r2_cutoff, test_size, random_state, save_cavs):
        """Compute R² score for image models."""
        return compute_r2_identifiability_score_image(
            loader=loader,
            embedder_model=embedder_model,
            config=self.config,
            concept_columns=concept_columns,
            r2_cutoff=r2_cutoff,
            test_size=test_size,
            random_state=random_state,
            save_cavs=save_cavs
        )
    
    def _compute_r2_multimodal(self, loader, embedder_model, processor,
                              concept_columns, r2_cutoff, test_size, random_state,
                              save_cavs, use_images):
        """Compute R² score for multimodal models."""
        if processor is None:
            raise ValueError("processor is required for multimodal models")
        
        return compute_r2_identifiability_score_multimodal(
            loader=loader,
            embedder_model=embedder_model,
            processor=processor,
            config=self.config,
            concept_columns=concept_columns,
            r2_cutoff=r2_cutoff,
            test_size=test_size,
            random_state=random_state,
            save_cavs=save_cavs,
            use_images=use_images
        )


    # ============================================================================
    # F1 IDENTIFIABILITY SCORE
    # ============================================================================
    
    def compute_identifiability_score(self,
                                         data_source: Union[pd.DataFrame, DataLoader],
                                         model,
                                         embedder_tokenizer=None,
                                         cavs: Optional[Dict] = None,
                                         f1_cutoff: Optional[float] = None,
                                         test_size: float = 0.3,
                                         random_state: int = 42,
                                         cos_cubed: bool = False,
                                         text_column: str = "text"):
        """
        Compute identifiability score
        
        IMPORTANT: 
        - For CB-LLM annotation, this uses R² regression instead of F1 thresholding
        - For other annotations, computes optimal thresholds to maximize F1 scores
        
        Args:
            data_source: DataFrame (for text/gemma) or DataLoader (for multimodal/image)
            model: Model for extracting embeddings
            embedder_tokenizer: Tokenizer (for text modes)
            cavs: Dictionary of CAV vectors
            f1_cutoff: Threshold for filtering concepts (None = keep all)
            test_size: Validation set proportion
            random_state: Random seed
            cos_cubed: Whether to cube cosine similarities
            text_column: Name of text column in DataFrame
        
        Returns:
            Tuple: (cosine_train, cosine_val, cosine_df, thresholds, metrics, filtered_concepts)
            For CB-LLM: (train_df, val_df, metrics, filtered_concepts)
        """
        print(f"📊 Computing F1 identifiability score in {self.modality_mode} mode...")
        
        # Special handling for CB-LLM: Use R² regression instead
        if self._is_cb_llm_annotation():
            print("🔷 CB-LLM annotation: Using R² regression method instead of F1 thresholding")
            return self._compute_R2_cb_llm(
                data_source, model, embedder_tokenizer,
                cavs, test_size, random_state
            )
        
        # Route to appropriate function
        if self.is_multimodal:
            return self._compute_f1_multimodal(
                data_source, model, embedder_tokenizer, cavs, f1_cutoff, cos_cubed
            )
        elif self.modality_mode == 'image':
            return self._compute_f1_image(
                data_source, model, cavs, f1_cutoff, cos_cubed
            )
        else:
            # Text-only (BERT, Gemma, or CLIP/BLIP text-only)
            return self._compute_f1_text(
                data_source, model, embedder_tokenizer, cavs,
                f1_cutoff, text_column, cos_cubed
            )
    
    def _compute_f1_text(self, df, model, embedder_tokenizer, cavs,
                        f1_cutoff, text_column, cos_cubed):
        """Compute F1 score for text/gemma models."""
        if embedder_tokenizer is None:
            raise ValueError("embedder_tokenizer is required for text models")
        
        # Import appropriate function
        if self.is_gemma:
            from new_heuristique_text import compute_cosine_matrix_and_metrics_gemma_version as compute_fn
        else:
            from new_heuristique_text import compute_cosine_matrix_and_metrics as compute_fn
        
        cosine_train, cosine_val, cosine_df, thresholds, metrics, filtered = compute_fn(
            df=df,
            text_column=text_column,
            model=model,
            embedder_tokenizer=embedder_tokenizer,
            cavs=cavs,
            f1_cutoff=f1_cutoff if f1_cutoff is not None else 0,
            device=self.config.device,
            save_dir=f"{self.config.SAVE_PATH}/blue_checkpoints/{self.config.model_name}/cavs/{self.config.cavs_type}",
            config=self.config,
            annotation=self.config.annotation,
            cos_cubed=cos_cubed
        )
        
        print(f"✓ F1 identifiability computed: {len(filtered)} concepts retained")
        return cosine_train, cosine_val, cosine_df, thresholds, metrics, filtered
    
    def _compute_f1_multimodal(self, loader, model, embedder_tokenizer, cavs, f1_cutoff, cos_cubed):
        """Compute F1 score for multimodal models."""
        from new_heuristique_multimodal import compute_cosine_matrix_and_metrics_multimodal_dataloader as compute_fn
        
        cosine_train, cosine_val, cosine_df, thresholds, metrics, filtered = compute_fn(
            dataloader=loader,
            model=model,
            tokenizer = embedder_tokenizer,
            cavs=cavs,
            f1_cutoff=f1_cutoff,
            device=self.config.device,
            save_dir=f"{self.config.SAVE_PATH}/blue_checkpoints/{self.config.model_name}/cavs/{self.config.cavs_type}",
            config=self.config,
            annotation=self.config.annotation,
            cos_cubed=cos_cubed
        )
        
        print(f"✓ F1 identifiability computed (multimodal): {len(filtered)} concepts retained")
        return cosine_train, cosine_val, cosine_df, thresholds, metrics, filtered
    
    def _compute_f1_image(self, loader, model, cavs, f1_cutoff, cos_cubed):
        """Compute F1 score for image models."""
        from new_heuristique_image import compute_cosine_matrix_and_metrics_image_dataloader as compute_fn
        
        cosine_train, cosine_val, cosine_df, thresholds, metrics, filtered = compute_fn(
            dataloader=loader,
            model=model,
            cavs=cavs,
            f1_cutoff=f1_cutoff,
            device=self.config.device,
            save_dir=f"{self.config.SAVE_PATH}/blue_checkpoints/{self.config.model_name}/cavs/{self.config.cavs_type}",
            config=self.config,
            annotation=self.config.annotation,
            cos_cubed=cos_cubed
        )
        
        print(f"✓ F1 identifiability computed (image): {len(filtered)} concepts retained")
        return cosine_train, cosine_val, cosine_df, thresholds, metrics, filtered
    
    def _compute_R2_cb_llm(self, df_cosine, model, embedder_tokenizer,
                          cavs, test_size, random_state):
        """
        Compute identifiability for CB-LLM using R² regression.
        CB-LLM doesn't use F1 thresholding, it uses continuous regression.
        
        CORRECTION: Utilise maintenant la signature unifiée avec dispatch automatique.
        """
        from identifiability_score_R2_unified import compute_r2_identifiability_score
        
        # Détermination du model_type
        model_type = 'gemma' if 'gemma' in self.config.model_name.lower() else 'bert'
        
        # Appel unifié avec les bons paramètres
        train_df, val_df, metrics, filtered, cavs_computed = compute_r2_identifiability_score(
            df_cosine=df_cosine,
            embedder_model=model,
            embedder_tokenizer=embedder_tokenizer,
            config=self.config,
            concept_columns= ['concept_' + str(a) for a in list(cavs.keys())] if cavs else None,
            r2_cutoff=None,  # Pas de filtrage pour CB-LLM
            test_size=test_size,
            random_state=random_state,
            save_cavs=True,
            modality_mode=self.modality_mode  # Utilise le mode détecté
        )
        
        print(f"✓ R² identifiability computed (CB-LLM): {len(filtered)} concepts retained")
        
        # Return in similar format but with None for items not applicable to CB-LLM
        return train_df, val_df, None, None, metrics, filtered
        
    # ============================================================================
    # COMBINED SCORES
    # ============================================================================
    
    def compute_combined_scores(self,
                                df_aug_train: pd.DataFrame,
                                top_n: int = 10,
                                save_results: bool = True) -> pd.DataFrame:
        """
        Calcule les scores combinés en fusionnant F1/R², TCAV, LIG et Présence.
        
        Vérifie automatiquement l'existence des fichiers requis et calcule :
        - combined_score_TCAVS = Identifiability × TCAV × Présence (si applicable)
        - combined_score_LIG = Identifiability × LIG_normalized
        
        Args:
            df_aug_train: DataFrame d'entraînement (pour calcul des scores de présence)
            top_n: Nombre de top concepts à afficher dans le résumé
            save_results: Si True, sauvegarde les résultats en CSV et JSON
        
        Returns:
            DataFrame avec tous les scores et scores combinés
        """
        print("=" * 80)
        print("🧮 CALCUL DES SCORES COMBINÉS")
        print("=" * 80)
        
        base_path = f"{self.config.SAVE_PATH}/blue_checkpoints/{self.config.model_name}/cavs/{self.config.cavs_type}"
        suffix = f"{self.config.annotation}_{self.config.agg_mode}_{self.config.agg_scope}"
        
        print(f"📂 Chemin : {base_path}")
        print(f"📝 Annotation : {self.config.annotation}")
        print(f"🔷 Mode CB-LLM : {self._is_cb_llm_annotation()}")
        
        # Chargement des scores
        identifiability_scores = self._load_identifiability_scores_for_combined(base_path)
        tcav_scores = self._load_tcav_scores_for_combined(base_path)
        lig_scores = self._load_lig_scores_for_combined(base_path)
        presence_scores = self._compute_presence_scores_for_combined(df_aug_train)

        # Nettoyage uniquement des espaces (pas de clean_concept_name)
        print("\n🧹 Nettoyage des espaces dans les noms de concepts...")
        identifiability_scores = self._clean_strings_dict(identifiability_scores)
        tcav_scores = self._clean_strings_dict(tcav_scores)
        lig_scores = self._clean_strings_dict(lig_scores)
        if presence_scores:
            presence_scores = self._clean_strings_dict(presence_scores)
        
        # Construction du DataFrame
        plot_df = self._build_combined_dataframe(
            identifiability_scores, tcav_scores, lig_scores, presence_scores
        )
        
        # Calcul des scores combinés
        plot_df = self._compute_combined_columns(plot_df)
        
        # Sauvegarde
        if save_results:
            self._save_combined_scores(plot_df, base_path, suffix)
        
        # Résumé
        self._print_combined_summary(plot_df, top_n)
        
        print("\n" + "=" * 80)
        print("✅ CALCUL DES SCORES COMBINÉS TERMINÉ")
        print("=" * 80)
        
        return plot_df
    
    def _load_identifiability_scores_for_combined(self, base_path: str) -> Dict[str, float]:
        """Charge les scores d'identifiabilité (F1 ou R²)."""
        print("\n📊 [1/4] Chargement des scores d'identifiabilité...")
        
        if self._is_cb_llm_annotation():
            # CB-LLM : R² scores
            file_path = os.path.join(
                base_path,
                f"r2_identifiability_{self.config.annotation}.json"
            )
            score_key = 'R2_val'
            score_type = "R²"
        else:
            # Autres : F1 scores
            file_path = os.path.join(
                base_path,
                f"detection_concept_{self.config.cavs_type}_{self.config.annotation}_{self.config.agg_mode}_{self.config.agg_scope}.json"
            )
            score_key = 'F1'
            score_type = "F1"
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"❌ Fichier d'identifiabilité introuvable : {file_path}\n"
                f"   Veuillez d'abord calculer les scores {score_type}."
            )
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        scores = {key: perf[score_key] for key, perf in data.items()}
        print(f"   ✓ {len(scores)} scores {score_type} chargés")
        return scores
    
    def _load_tcav_scores_for_combined(self, base_path: str) -> Dict[str, float]:
        """Charge les scores TCAV."""
        print("\n📊 [2/4] Chargement des scores TCAV...")
        
        file_path = os.path.join(base_path, f"TCAVS_scores_{self.config.annotation}.json")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"❌ Fichier TCAV introuvable : {file_path}\n"
                f"   Veuillez d'abord calculer les scores TCAV."
            )
        
        with open(file_path, 'r') as f:
            tcav_data = json.load(f)
        
        # Conversion en dict
        if isinstance(tcav_data, list):
            scores = {item[0]: item[1] for item in tcav_data}
        elif isinstance(tcav_data, dict):
            scores = tcav_data
        else:
            raise ValueError(f"Format TCAV non reconnu : {type(tcav_data)}")
        
        print(f"   ✓ {len(scores)} scores TCAV chargés")
        return scores
    
    def _load_lig_scores_for_combined(self, base_path: str) -> Dict[str, float]:
        """Charge les scores LIG."""
        print("\n📊 [3/4] Chargement des scores LIG...")
        
        file_path = os.path.join(
            base_path,
            f"LIG_scores_{self.config.cavs_type}_{self.config.annotation}_{self.config.agg_mode}_{self.config.agg_scope}.json"
        )
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"❌ Fichier LIG introuvable : {file_path}\n"
                f"   Veuillez d'abord calculer les scores LIG."
            )
        
        with open(file_path, 'r') as f:
            lig_data = json.load(f)
        
        # Conversion en dict (sans clean_concept_name)
        if isinstance(lig_data, list):
            # Supprimer les doublons
            lig_data = list(dict.fromkeys(tuple(item) for item in lig_data))
            scores = {item[0]: item[1] for item in lig_data}
        elif isinstance(lig_data, dict):
            scores = lig_data
        else:
            raise ValueError(f"Format LIG non reconnu : {type(lig_data)}")
        
        print(f"   ✓ {len(scores)} scores LIG chargés")
        return scores
    
    def _compute_presence_scores_for_combined(self, df_aug_train: pd.DataFrame) -> Optional[Dict[str, float]]:
        """Calcule les scores de présence."""
        print("\n📊 [4/4] Calcul des scores de présence...")
        
        if df_aug_train is None:
            print("   ⚠️  df_aug_train non fourni, scores de présence ignorés")
            return None
        
        # Colonnes à exclure
        exclude_cols = ['text', 'label', 'Unnamed: 0', 'abstract', 'image_id', 'section']
        
        # Debug : afficher les colonnes disponibles
        all_cols = df_aug_train.columns.tolist()
        print(f"   🔍 Colonnes dans df_aug_train : {all_cols[:10]}...")  # Affiche les 10 premières
        
        # Pour CB-LLM : moyenne des valeurs (colonnes concept_*)
        # Pour autres : proportion de 1 (colonnes concept_* ou dummy_*)
        presence_scores = {}
        
        for concept_col in df_aug_train.columns:
            if concept_col in exclude_cols:
                continue
            
            # Extraire le nom du concept (en gérant plusieurs cas)
            concept_name = None
            if concept_col.startswith('concept_'):
                concept_name = concept_col.replace('concept_', '')
            elif concept_col.startswith('dummy_'):
                concept_name = concept_col.replace('dummy_', '')
            elif concept_col.startswith('cos_'):
                concept_name = concept_col.replace('cos_', '')
            else:
                # Si pas de préfixe connu, utiliser le nom tel quel
                # (en excluant les colonnes métadata)
                if concept_col not in exclude_cols:
                    concept_name = concept_col
            
            if concept_name is None:
                continue
            
            # Calcul du score selon le mode
            try:
                if self._is_cb_llm_annotation():
                    # CB-LLM : moyenne des valeurs (valeurs continues)
                    presence_scores[concept_name] = df_aug_train[concept_col].mean()
                else:
                    # Autres : proportion de 1 (valeurs binaires)
                    presence_scores[concept_name] = df_aug_train[concept_col].sum() / len(df_aug_train)
            except Exception as e:
                print(f"   ⚠️  Erreur pour colonne '{concept_col}': {e}")
                continue
        
        if presence_scores:
            mode_str = "moyenne des valeurs" if self._is_cb_llm_annotation() else "proportion de 1"
            print(f"   ✓ {len(presence_scores)} scores de présence calculés ({mode_str})")
            # Debug : afficher quelques scores
            sample_scores = list(presence_scores.items())[:5]
            print(f"   🔍 Exemples : {sample_scores}")
        else:
            print("   ⚠️  Aucun score de présence calculé")
            print(f"   🔍 Colonnes exclues : {exclude_cols}")
            return None
        
        return presence_scores

    @staticmethod
    def _clean_strings_dict(scores: Dict[str, float]) -> Dict[str, float]:
        """Nettoie uniquement les espaces dans les clés."""
        import re
        return {re.sub(r'\s+', ' ', k.strip()): v for k, v in scores.items()}
    
    def _build_combined_dataframe(self, identifiability_scores, tcav_scores, 
                                   lig_scores, presence_scores) -> pd.DataFrame:
        """Construit le DataFrame avec tous les scores."""
        print("\n📊 Construction du DataFrame...")
        
        score_name = "R² Score" if self._is_cb_llm_annotation() else "F1 Score detected"
        
        df_data = {
            score_name: identifiability_scores,
            "TCAVS score": tcav_scores,
            "LIG score": lig_scores,
        }
        
        if presence_scores is not None:
            df_data["Presence (Mean)"] = presence_scores
        
        plot_df = pd.DataFrame(df_data)
        plot_df['concept'] = plot_df.index
        
        print(f"   ✓ DataFrame créé : {plot_df.shape}")
        return plot_df
    
    def _compute_combined_columns(self, plot_df: pd.DataFrame) -> pd.DataFrame:
        """Calcule les colonnes de scores combinés."""
        print("🧮 Calcul des scores combinés...")
        
        score_name = "R² Score" if self._is_cb_llm_annotation() else "F1 Score detected"
        
        # Valeur absolue AVANT tout calcul
        plot_df[score_name] = plot_df[score_name].abs()
        
        # Min-Max scaling pour ramener dans [0, 1]
        score_values = plot_df[score_name].values
        score_min = score_values.min()
        score_max = score_values.max()
        
        if score_max > score_min:
            plot_df[score_name] = (score_values - score_min) / (score_max - score_min)
        else:
            # Toutes les valeurs sont identiques
            plot_df[score_name] = 0.5
        
        # Normalisation du score LIG
        lig_values = plot_df["LIG score"].values
        lig_min = lig_values.min()
        lig_max = lig_values.max()
        
        if lig_max == lig_min:
            plot_df["LIG score norm"] = 1.0
        else:
            plot_df["LIG score norm"] = (lig_values - lig_min) / (lig_max - lig_min)
        
        # Score combiné TCAVS et LIG
        if "Presence (Mean)" in plot_df.columns:
            # Score combiné TCAVS
            plot_df["combined_score_TCAVS"] = (
                plot_df[score_name] * plot_df["TCAVS score"] 
            )
            # plot_df["combined_score_TCAVS"] = (
            #     plot_df[score_name] * plot_df["TCAVS score"] * plot_df["Presence (Mean)"]
            # )
            
            # Score combiné LIG
            # correction apportée
            # plot_df["combined_score_LIG"] = plot_df[score_name] * plot_df["LIG score norm"] * plot_df["Presence (Mean)"]
            plot_df["combined_score_LIG"] = plot_df[score_name] * plot_df["LIG score norm"] 
        else:
            # Score combiné TCAVS
            plot_df["combined_score_TCAVS"] = plot_df[score_name] * plot_df["TCAVS score"]
            # Score combiné LIG
            plot_df["combined_score_LIG"] = plot_df[score_name] * plot_df["LIG score norm"]
            
        print(f"   ✓ Scores combinés calculés")
        return plot_df
        
    
    def _save_combined_scores(self, plot_df: pd.DataFrame, base_path: str, suffix: str) -> None:
        """Sauvegarde les résultats."""
        print("\n💾 Sauvegarde des résultats...")
        
        os.makedirs(base_path, exist_ok=True)
        
        # CSV complet
        csv_path = os.path.join(base_path, f"combined_score_concept_{suffix}.csv")
        plot_df.to_csv(csv_path, index=False)
        print(f"   ✓ CSV : combined_score_concept_{suffix}.csv")
        
        # JSON TCAVS (trié)
        combined_tcavs = dict(zip(
            plot_df.sort_values('combined_score_TCAVS', ascending=False)['concept'],
            plot_df.sort_values('combined_score_TCAVS', ascending=False)['combined_score_TCAVS']
        ))
        tcavs_path = os.path.join(base_path, f"combined_score_TCAVS_{suffix}.json")
        with open(tcavs_path, 'w') as f:
            json.dump(combined_tcavs, f, ensure_ascii=False, indent=4)
        print(f"   ✓ JSON : combined_score_TCAVS_{suffix}.json")
        
        # JSON LIG (trié)
        combined_lig = dict(zip(
            plot_df.sort_values('combined_score_LIG', ascending=False)['concept'],
            plot_df.sort_values('combined_score_LIG', ascending=False)['combined_score_LIG']
        ))
        lig_path = os.path.join(base_path, f"combined_score_LIG_{suffix}.json")
        with open(lig_path, 'w') as f:
            json.dump(combined_lig, f, ensure_ascii=False, indent=4)
        print(f"   ✓ JSON : combined_score_LIG_{suffix}.json")

    def _print_combined_summary(self, plot_df: pd.DataFrame, top_n: int) -> None:
        """Affiche un résumé des résultats."""
        print("\n" + "=" * 80)
        print("📊 RÉSUMÉ DES SCORES COMBINÉS")
        print("=" * 80)
        
        score_name = "R² Score" if self._is_cb_llm_annotation() else "F1 Score detected"
        
        print(f"\n📈 Statistiques :")
        print(f"   - Nombre de concepts : {len(plot_df)}")
        print(f"   - Note : {score_name} est normalisé [0,1] (min-max) après valeur absolue")
        
        for col in [score_name, "TCAVS score", "LIG score norm"]:
            if col in plot_df.columns:
                mean_val = plot_df[col].mean()
                min_val = plot_df[col].min()
                max_val = plot_df[col].max()
                print(f"   - {col:20s} : mean={mean_val:.3f}, min={min_val:.3f}, max={max_val:.3f}")
        
        # Top TCAVS
        print(f"\n🏆 Top {top_n} concepts (Combined TCAVS) :")
        top_tcavs = plot_df.nlargest(top_n, 'combined_score_TCAVS')
        for i, (_, row) in enumerate(top_tcavs.iterrows(), 1):
            print(f"   {i:2d}. {row['concept']:30s} → combined={row['combined_score_TCAVS']:.4f} "
                  f"({score_name}={row[score_name]:.3f}, TCAVS={row['TCAVS score']:.3f})")
        
        # Top LIG
        print(f"\n🏆 Top {top_n} concepts (Combined LIG) :")
        top_lig = plot_df.nlargest(top_n, 'combined_score_LIG')
        for i, (_, row) in enumerate(top_lig.iterrows(), 1):
            print(f"   {i:2d}. {row['concept']:30s} → combined={row['combined_score_LIG']:.4f} "
                  f"({score_name}={row[score_name]:.3f}, LIG={row['LIG score norm']:.3f})")
        
        print("=" * 80)

    # ============================================================================
    #  SCORES PAR COVERGAGE
    # ============================================================================

    def coverage_analysis(self, df_aug_train: pd.DataFrame, TCAVS_or_LIG):
        """
        Point d'entrée pour l'analyse de coverage depuis le dispatcher principal.

        Returns:
            Résultats de l'analyse de coverage

        """
        # Après Step 5 (scoring)
        if not self.config.cb_llm_mode:
            from concept_coverage_analysis import do_coverage_analysis
        
            return do_coverage_analysis(
                config=self.config,
                df_aug_train=df_aug_train,
                TCAVS_or_LIG = TCAVS_or_LIG
            )
        else: 
            return print("not possible to do coverage analysis with cb_llm annotation") 

    
    # ============================================================================
    # UTILITY METHODS
    # ============================================================================
    
    def get_mode_info(self) -> Dict[str, Any]:
        """
        Get information about current modality mode.
        
        Returns:
            Dict with mode information
        """
        return {
            'modality_mode': self.modality_mode,
            'model_name': self.config.model_name,
            'dataset': self.config.DATASET,
            'annotation': getattr(self.config, 'annotation', 'unknown'),
            'is_cb_llm': self._is_cb_llm_annotation(),
            'is_gemma': self.is_gemma,
            'is_multimodal': self.is_multimodal,
            'is_clip_text': self.is_clip_text,
            'uses_images': self.modality_mode in ["multimodal", "image"],
            'uses_text': self.modality_mode in ["multimodal", "text"],
            'supported_operations': self.get_supported_operations()
        }
    
    def get_supported_operations(self) -> Dict[str, bool]:
        """
        Get information about which operations are supported for current annotation.
        
        Returns:
            Dict with operation support flags
        """
        if self._is_cb_llm_annotation():
            return {
                'mean_minus_cavs': False,
                'r2_cavs': True,
                'tcav_scores': False,  # Not recommended
                'lig_ranking': True,
                'clustering': False
            }
        else:
            return {
                'mean_minus_cavs': True,
                'r2_cavs': True,
                'tcav_scores': True,
                'lig_ranking': True,
                'clustering': True
            }
    
    def __repr__(self) -> str:
        return (f"ModalityDispatcher(mode={self.modality_mode}, "
                f"model={self.config.model_name}, dataset={self.config.DATASET})")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_dispatcher(model_name: str, 
                     dataset: str, 
                     annotation: str = 'C3M', 
                     **kwargs) -> ModalityDispatcher:
    """
    Quick dispatcher creation with minimal config.
    
    Args:
        model_name: Name of the model (e.g., 'clip', 'bert-base-uncased')
        dataset: Dataset name (e.g., 'n24news', 'agnews')
        annotation: Annotation type (default: 'C3M')
        **kwargs: Additional config parameters
    
    Returns:
        Configured ModalityDispatcher instance
    
    Example:
        >>> dispatcher = create_dispatcher('clip', 'n24news', annotation='C3M')
        >>> data = dispatcher.load_data()
    """
    from unified_config import load_config
    
    config = load_config(model_name, dataset)
    config.annotation = annotation
    
    # Apply additional kwargs to config
    for key, value in kwargs.items():
        setattr(config, key, value)
    
    return ModalityDispatcher(config)