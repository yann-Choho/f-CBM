import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
import torch.nn.functional as F
from torchvision import transforms, datasets
from tqdm import tqdm
import pandas as pd
import os
import copy
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy, ttest_ind
import numpy as np
import glob
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score, classification_report, mean_squared_error
from transformers import BlipProcessor, BlipModel, AutoProcessor
from transformers import CLIPProcessor, CLIPModel, AutoProcessor
from sklearn.metrics.pairwise import cosine_similarity
import re
from functools import lru_cache
import multiprocessing
from scipy.stats import pearsonr, spearmanr
import mmap
import json
import warnings
import random
import math
from torch import Tensor
from data_N24 import concepts_N24
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import networkx as nx
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import cosine_similarity

from data_N24_concepts import find_class_from_concept

from KAN import SimpleKAN, KANLinear, visualize_learned_functions

import matplotlib.pyplot as plt

##################################################################################################
# from https://github.com/AngelosNal/PyTorch-Gumbel-Sigmoid/blob/main/gumbel_sigmoid.py

def gumbel_sigmoid(logits: Tensor, tau: float = 0.5, hard: bool = True, threshold: float = 0.5) -> Tensor:
    """
    Samples from the Gumbel-Sigmoid distribution and optionally discretizes.
    The discretization converts the values greater than `threshold` to 1 and the rest to 0.
    The code is adapted from the official PyTorch implementation of gumbel_softmax:
    https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#gumbel_softmax

    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized,
            but will be differentiated as if it is the soft sample in autograd
     threshold: threshold for the discretization,
                values greater than this will be set to 1 and the rest to 0

    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Sigmoid distribution.
      If ``hard=True``, the returned samples are descretized according to `threshold`, otherwise they will
      be probability distributions.

    """

    #gumbels1 = (-torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log())  # ~Gumbel(0, 1)
    #gumbels2 = (-torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log())  # ~Gumbel(0, 1)
    gumbels1 = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
    gumbels2 = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
    gumbels = (logits + gumbels1 - gumbels2) / tau  # ~Gumbel(logits, tau)
    y_soft = gumbels.sigmoid()

    if hard:
        # Straight through.
        indices = (y_soft > threshold).nonzero(as_tuple=True)

        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format)

        if len(indices) == 1:
            # 1D case
            y_hard[indices[0]] = 1.0
        elif len(indices) == 2:
            # 2D case
            y_hard[indices[0], indices[1]] = 1.0

        ret = y_hard - y_soft.detach() + y_soft
        if(hard==True):
            return ret
        elif(hard=='both'):
            return ret, y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
        return ret

##################################################################################################
# inspired from https://arxiv.org/abs/2504.18026v3 https://github.com/Emilianopp/direct-preference-optimization

def cpo_loss(logits, empirical_concepts, temperature=1.0, num_samples=5, threshold=0.5):
    """
    logits: Tensor of shape (batch_size, num_concepts)
    empirical_concepts: Tensor of shape (batch_size, num_concepts), binary (0/1)
    temperature: float, temperature for Gumbel-Sigmoid sampling
    num_samples: int, number of samples per input for expectation
    """
    batch_size, num_concepts = logits.shape

    # Calculate log prob for each concept individually
    log_prob_empirical = -F.binary_cross_entropy_with_logits(
        logits, empirical_concepts.float(), reduction='none'
    )  # Keep shape: (batch_size, num_concepts)

    # Sample c' from the model's policy using Gumbel-Sigmoid for differentiability
    concept_probs = torch.sigmoid(logits)

    total_loss = 0.0
    valid_samples = 0

    for _ in range(num_samples):
        # Gumbel-Sigmoid sampling for each concept
        gumbel_noise1 = -torch.log(-torch.log(torch.rand_like(concept_probs) + 1e-8) + 1e-8)
        gumbel_noise2 = -torch.log(-torch.log(torch.rand_like(concept_probs) + 1e-8) + 1e-8)
        sampled_logits = (logits + gumbel_noise1 - gumbel_noise2) / temperature
        y_soft = torch.sigmoid(sampled_logits)
        
        # Hard sampling with straight-through
        y_hard_mask = (y_soft > threshold)
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format)
        y_hard[y_hard_mask] = 1.0
        
        sampled_concepts = y_hard - y_soft.detach() + y_soft

        # Concept-wise: loss applied only to concepts that differ
        concept_mask = (sampled_concepts != empirical_concepts)  # (batch_size, num_concepts)

        if concept_mask.sum() > 0:
            # Only penalize the specific concepts that were sampled incorrectly
            total_loss += -log_prob_empirical[concept_mask].sum()
            valid_samples += concept_mask.sum().item()

    if valid_samples == 0:
        # No mismatched samples, return zero loss
        return torch.tensor(0.0, requires_grad=True, device=logits.device)

    return total_loss / valid_samples

##################################################################################################


def cos_sim_cubed(cbl_features, target):
    """
    cbl_features: [batch_size, n_concepts]
    target: [batch_size, n_concepts]
    Computes similarity per sample, then averages
    """
    # introduced in LABEL-FREE CONCEPT BOTTLENECK MODELS
    # also used in CB-LLM
    
    # Centrage (mean centering)
    # Shapes: [batch_size, n_concepts]
    cbl_features = cbl_features - torch.mean(cbl_features, dim=-1, keepdim=True)
    # dim=-1 is the concept dimension
    # Computes mean across concepts for each sample
    # Result: [batch_size, 1]
    # Centers each sample's concept values (makes mean=0 for each sample)

    target = target - torch.mean(target, dim=-1, keepdim=True)
    # Same: [batch_size, 1]

    # Cube and normalize along concept dimension
    cbl_features = F.normalize(cbl_features**3, dim=-1)
    # Cubes each value, then normalizes each sample's concept vector to unit norm
    # Result: [batch_size, n_concepts]

    target = F.normalize(target**3, dim=-1)
    # Result: [batch_size, n_concepts]

    # Compute dot product along concept dimension
    sim = torch.sum(cbl_features * target, dim=-1)
    # Element-wise multiplication then sum across concepts
    # Result: [batch_size] - one similarity score per sample

    return sim.mean()
    # Result: scalar - average similarity across samples

##################################################################################################
class CBM_model(nn.Module):
    def __init__(self, device, class_dict, base_model, processor, num_classes, concept_list, 
                 concept_training=False, frozen_backbone=False, concept_representation='gumbel_sigmoid', multimodal_features=False,
                 lambda_XtoC=0.5, lambda_XtoC_leakage=0.5, alpha=0.01, l1_ratio=0.5, relu_concepts=False, cpo=False, contrastive=False, 
                 pos_weight=True, loss_rescaled=True, leakage_loss=False, leakage_loss_activation="up", loss_CBLLM='MSE', kan_layer=False):
        """
        Initialize the CBM model.
        
        Args:
            device: The device to run the model on
            class_dict: Dictionary mapping class indices to class names
            base_model: Base BLIP model
            processor: BLIP processor
            num_classes: Number of output classes
            concept_list: List of concept names
            concept_training: Whether to train with concept supervision
            frozen_backbone: Whether to freeze the backbone model
            concept_representation: How to represent concepts for the classifier
                - 'gumbel_sigmoid': Use Gumbel sigmoid for hard binary values
                - 'sigmoid': Use regular sigmoid for soft values
                - 'logits': Use the raw logits directly
                - 'importance': Use raw logits as concept activations correspond to importance scores ranging from -1 to 1
            multimodal_features: Whether to use multimodal features
                - only available with BLIP (never tested)
            lambda_XtoC: Weight for the XtoC loss
            alpha: Weight for the L1 regularization
            l1_ratio: Ratio of L1 regularization
            relu_concepts: Whether to apply ReLU activation to concepts before passing to classifier
            cpo: Whether to use CPO loss for concept training https://arxiv.org/abs/2504.18026v3
        """
        super(CBM_model, self).__init__()
        self.base_model = base_model
        self.processor = processor
        self.class_dict = class_dict
        self.concept_list = list(concept_list)
        self.device = device
        self.num_classes = num_classes
        self.num_concepts = len(self.concept_list) # separated by modality in case combine='concat'
        self.multimodal_features = multimodal_features
        self.contrastive = contrastive
        self.temperature = 0.07
        self.pos_weight = pos_weight
        self.loss_rescaled = loss_rescaled
        self.leakage_loss = leakage_loss
        self.leakage_loss_activation = leakage_loss_activation
        self.loss_CBLLM = loss_CBLLM
        self.kan_layer = kan_layer

        # automatically check if text & image concepts are present in dataset
        # self.combine_type = text (only text concepts)
        # self.combine_type = image (only image concepts)
        # self.combine_type = concat (both text and image concepts)
        # self.combine_type = combine (text & image concepts combined)
        has_image_concept = np.sum([True if 'concept_image_' in concept else False for concept in self.concept_list])>0
        has_text_concept = np.sum([True if 'concept_text_' in concept else False for concept in self.concept_list])>0
        if(has_image_concept and has_text_concept):
            self.combine_type = 'concat'
        elif(has_image_concept):
            self.combine_type = 'image'
        elif(has_text_concept):
            self.combine_type = 'text'
        else:
            self.combine_type = 'combine'

        print('has image concept:',has_image_concept, ' has text concept:',has_text_concept, ' combine type:',self.combine_type)

        if(self.combine_type=='concat'):
            # make sure concept list is in the right order
            # we want to have first text concepts, and then image concepts, in the same concept order
            # because of the way we are combining them in function _or_concept_layer
            unique_concepts = np.unique([concept.replace('concept_text_','').replace('concept_image_','') for concept in self.concept_list])
            self.concept_list = ['concept_text_'+concept for concept in unique_concepts] + ['concept_image_'+concept for concept in unique_concepts]
            if(len(self.concept_list)!=self.num_concepts):
                raise ValueError("Concept set is not complete, should have same list of concepts for both modalities")

        # define linear layers for concept & task predictions

        if(self.multimodal_features or self.combine_type=='text' or self.combine_type=='image'):
            # if multimodal features are used (so one vector for both modalities, only available with BLIP)
            # or if only text or image concepts are used, then just one vector as output from CLIP/BLIP
            self.concept_layer = nn.Linear(base_model.config.projection_dim, self.num_concepts)
            if(not self.kan_layer):
                self.classifier = nn.Linear(self.num_concepts, num_classes)
            else:
                self.classifier = KANLinear(self.num_concepts, num_classes) #SimpleKAN(input_dim=self.num_concepts, hidden_dim=0, output_dim=num_classes)

        elif(self.combine_type=='combine'):
            # text and image embeddings are produced separately, and are combined into one concept
            self.concept_layer = nn.Linear(base_model.config.projection_dim*2, self.num_concepts)
            if(not self.kan_layer):
                self.classifier = nn.Linear(self.num_concepts, num_classes)
            else:
                self.classifier = KANLinear(self.num_concepts, num_classes) # SimpleKAN(input_dim=self.num_concepts, hidden_dim=0, output_dim=num_classes)

        elif(self.combine_type=='concat'):
            # text and image embeddings are produced separately, and predict image & text concept separately
            # here instead, we need one layer connecting text embeddings produced by clip to text concepts
            # and one layer connecting image embeddings produced by clip to image concepts
            self.concept_layer_text = nn.Linear(base_model.config.projection_dim, int(self.num_concepts/2))
            self.concept_layer_image = nn.Linear(base_model.config.projection_dim, int(self.num_concepts/2))
            # and then we need to combine the two concepts (image & text) into one with the formula a+b-a*b
            #self.concept_layer = self._or_concept_layer
            if(not self.kan_layer):
                self.classifier = nn.Linear(int(self.num_concepts/2), num_classes)
            else:
                self.classifier = KANLinear(int(self.num_concepts/2), num_classes) # SimpleKAN(input_dim=int(self.num_concepts/2), hidden_dim=0, output_dim=num_classes)

        if(self.multimodal_features and (self.combine_type=='concat')):
            raise ValueError("Cannot use 'concat' combine_type with multimodal_features=True, #use 'combine' instead")

        self.concept_training = concept_training
        self.frozen_backbone = frozen_backbone
        self.concept_representation = concept_representation
        if(self.concept_representation=='importance'):
            self.pos_weight = False
        self.relu_concepts = relu_concepts
        self.cpo = cpo

        # Validate concept representation choice
        valid_representations = ['gumbel_sigmoid', 'sigmoid', 'logits', 'importance']
        if self.concept_representation not in valid_representations:
            raise ValueError(f"concept_representation must be one of {valid_representations}")

        # Freeze the CLIP/BLIP model
        if(self.frozen_backbone):
            # freeze all parameters
            print('freeze backbone')
            for param in self.base_model.parameters():
                param.requires_grad = False

            # unfreeze the last layer only
            if(self.frozen_backbone=='not_last_layer'):
                print('but not last layer')
                for name, param in self.base_model.named_parameters():
                    if ("visual_proj" in name) or ("text_proj" in name):
                        print(name)
                        param.requires_grad = True

        # training parameters
        self.lambda_XtoC = lambda_XtoC
        self.lambda_XtoC_leakage = lambda_XtoC_leakage
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        
    def _process_concept_logits(self, concept_logits, tau=0.5):
        """
        Process concept logits based on the configured representation mode
        
        Args:
            concept_logits: Raw concept logits from the concept layer
            tau: Temperature for Gumbel sigmoid
            
        Returns:
            The processed concept representation to pass to the classifier
        """
        if (self.concept_representation == 'gumbel_sigmoid'):
            return gumbel_sigmoid(concept_logits, tau=tau, hard=True, threshold=0.5)
        elif (self.concept_representation == 'sigmoid'):
            return torch.sigmoid(concept_logits)
        elif (self.concept_representation == 'logits') or (self.concept_representation == 'importance'):
            if(self.relu_concepts):
                concept_logits = torch.relu(concept_logits)
            return concept_logits
        else:
            raise ValueError(f"Unknown concept_representation: {self.concept_representation}")
        
    def _get_binary_concepts(self, concept_logits, tau=0.5):
        """
        Get binary concept predictions regardless of the concept representation mode
        Used for evaluation and metrics calculation
        
        Args:
            concept_logits: Raw concept logits from the concept layer
            
        Returns:
            Binary concept predictions (0 or 1)
        """
        if self.concept_representation == 'gumbel_sigmoid':
            # Already binary
            return gumbel_sigmoid(concept_logits, tau=tau, hard=True, threshold=0.5)
        else:
            # Convert to binary
            concept_logits = torch.as_tensor(concept_logits)
            return (concept_logits > 0.).float()
    
    def _or_concept_layer(self, concept_text_logits, concept_image_logits, tau=0.5):
        """
        Double concept layer that combines text and image concepts using the OR formula into a combined concept activation
        Outputs multimodal concept logits (for training), multimodal concept representations (for metrics or leakage) & combined concept representation for final head classifier
        """

        # for training the CBM
        combined_logits_multi = torch.cat((concept_text_logits, concept_image_logits), dim=1) 

        if(self.concept_representation=='logits' or self.concept_representation=='sigmoid' or self.concept_representation=='importance'):
            # Convert logits to probabilities
            concept_text_probs = torch.sigmoid(concept_text_logits)
            concept_image_probs = torch.sigmoid(concept_image_logits)
        elif(self.concept_representation=='gumbel_sigmoid'):
            # convert to binary
            concept_text_probs = gumbel_sigmoid(concept_text_logits, tau=tau, hard=True, threshold=0.5)
            concept_image_probs = gumbel_sigmoid(concept_image_logits, tau=tau, hard=True, threshold=0.5)

        # intermediate multimodal concept representation
        if((self.concept_representation=='logits') or (self.concept_representation=='importance')):
            combined_repr_multi = combined_logits_multi # representation as it is, just logits
        elif((self.concept_representation=='sigmoid') or (self.concept_representation=='gumbel_sigmoid')):
            combined_repr_multi = torch.cat((concept_text_probs, concept_image_probs), dim=1) # representation given by sigmoid or gumbel

        # Apply OR formula element-wise on probabilities
        combined_probs = concept_text_probs + concept_image_probs - concept_text_probs * concept_image_probs

        # convert back to logits when not gumbel or sigmoid
        if(self.concept_representation=='logits' or self.concept_representation=='importance'):
            combined_probs = torch.logit(combined_probs.clamp(min=1e-7, max=1-1e-7))

        return combined_logits_multi, combined_repr_multi, combined_probs

    def forward(self, input_ids, attention_mask, pixel_values, tau=0.5):

        if(not self.contrastive):

            if(not self.multimodal_features):
                outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
                if(self.combine_type=='text'):
                    combined_features = outputs.text_embeds
                elif(self.combine_type=='image'):
                    combined_features = outputs.image_embeds
                elif(self.combine_type=='combine'):
                    image_features = outputs.image_embeds
                    text_features = outputs.text_embeds
                    combined_features = torch.cat((text_features, image_features), dim=1) # use both modalities to predict combined concept
                elif(self.combine_type=='concat'):
                    image_features = outputs.image_embeds
                    text_features = outputs.text_embeds
            else:
                combined_features = self.base_model.get_multimodal_features(input_ids=input_ids, pixel_values=pixel_values,attention_mask=attention_mask)
                print('combined_features',combined_features.shape)
                combined_features = combined_features / combined_features.norm(dim=-1, keepdim=True)

            if(self.combine_type!='concat'):
                # case with just one modality (or combined concept modalities)
                concept_logits = self.concept_layer(combined_features)
            else:
                # in case of concat, two concept modalities are needed separately for training, and then combine them for final layer
                # First, get logits per modality
                concept_logits_texts = self.concept_layer_text(text_features)
                concept_logits_images = self.concept_layer_image(image_features)

        else:
            
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)

            # produce text embeddings associated to each concept
            if(self.combine_type!='concat'):
                text_labels = [re.sub(r"[^a-zA-Z\s]", "", label.replace('image_', '').replace('text_', '').replace('concept_', '').replace('_', ' ').replace('::', ' ')) for label in self.concept_list]
                text_inputs = self.processor(text=text_labels, return_tensors="pt", padding=True, truncation=True).to(self.device)
                comparison_embeddings = self.base_model.text_projection(self.base_model.text_model(**text_inputs).pooler_output)
                comparison_embeddings = F.normalize(comparison_embeddings, p=2, dim=1)
            else:
                text_labels_text = [re.sub(r"[^a-zA-Z\s]", "", label.replace('image_', '').replace('text_', '').replace('concept_', '').replace('_', ' ').replace('::', ' ')) for label in self.concept_list if 'concept_text_' in label]
                text_labels_image = [re.sub(r"[^a-zA-Z\s]", "", label.replace('image_', '').replace('text_', '').replace('concept_', '').replace('_', ' ').replace('::', ' ')) for label in self.concept_list if 'concept_image_' in label]
                text_inputs_text = self.processor(text=text_labels_text, return_tensors="pt", padding=True, truncation=True).to(self.device)
                text_inputs_image = self.processor(text=text_labels_image, return_tensors="pt", padding=True, truncation=True).to(self.device)
                comparison_embeddings_text = self.base_model.text_projection(self.base_model.text_model(**text_inputs_text).pooler_output)
                comparison_embeddings_image = self.base_model.text_projection(self.base_model.text_model(**text_inputs_image).pooler_output)
                comparison_embeddings_text = F.normalize(comparison_embeddings_text, p=2, dim=1)
                comparison_embeddings_image = F.normalize(comparison_embeddings_image, p=2, dim=1)

            if(self.combine_type=='text'):
                text_embeddings = F.normalize(outputs.text_embeds, p=2, dim=1)
                concept_logits = torch.matmul(text_embeddings, comparison_embeddings.t()) / self.temperature
            elif(self.combine_type=='image'):
                image_embeddings = F.normalize(outputs.image_embeds, p=2, dim=1)
                concept_logits = torch.matmul(image_embeddings, comparison_embeddings.t()) / self.temperature
            elif(self.combine_type=='combine'):
                text_embeddings = F.normalize(outputs.text_embeds, p=2, dim=1)
                image_embeddings = F.normalize(outputs.image_embeds, p=2, dim=1)
                concept_logits_texts = torch.matmul(text_embeddings, comparison_embeddings.t()) / self.temperature
                concept_logits_images = torch.matmul(image_embeddings, comparison_embeddings.t()) / self.temperature
                # 'OR' formula to combine concept probabilities
                concept_combined_probs = torch.sigmoid(concept_logits_images) + torch.sigmoid(concept_logits_texts) - torch.sigmoid(concept_logits_images) * torch.sigmoid(concept_logits_texts)  # element wise product
                # convert back to raw logits
                concept_logits = torch.logit(concept_combined_probs.clamp(min=1e-7, max=1-1e-7))
            elif(self.combine_type=='concat'):
                text_embeddings = F.normalize(outputs.text_embeds, p=2, dim=1)
                image_embeddings = F.normalize(outputs.image_embeds, p=2, dim=1)
                concept_logits_texts = torch.matmul(text_embeddings, comparison_embeddings_text.t()) / self.temperature
                concept_logits_images = torch.matmul(image_embeddings, comparison_embeddings_image.t()) / self.temperature

        # concept layer
        if(self.combine_type!='concat'):
            # Process concept logits based on representation mode
            concept_repr = self._process_concept_logits(concept_logits, tau=tau)

            # concept_repr_combined same as concept_repr
            concept_repr_combined = concept_repr

            # Forward to classifier
            logits = self.classifier(concept_repr)
        else:
            # Apply 'OR' operation to logits in order to combine them
            concept_logits, concept_repr, concept_repr_combined = self._or_concept_layer(concept_text_logits=concept_logits_texts, concept_image_logits=concept_logits_images, tau=tau)

            logits = self.classifier(concept_repr_combined)
        
        # return concept logits, concept representations, concept representations with multimodalities combined (in case of combine_type='concat'), and task logits
        return concept_logits, concept_repr, concept_repr_combined, logits
    
    # def predict(self, input_ids, attention_mask, pixel_values):
    #     self.to(self.device)
    #     self.eval()  # Set the model to evaluation mode
        
    #     with torch.no_grad():  # Disable gradient computation
    #         concept_logits, concept_repr, logits = self.forward(input_ids, attention_mask, pixel_values, tau=0.5)
            
    #         concept_binary = self.get_binary_concepts(concept_logits, tau=0.5)

    #         # Apply softmax to get probabilities
    #         probabilities = F.softmax(logits, dim=1)
            
    #         # Get the predicted class (index of the highest probability)
    #         predicted_classes = torch.argmax(probabilities, dim=1)
        
    #     # Return as a list for easier handling
    #     return [label for label in predicted_classes], probabilities, concept_binary, concept_logits
    
    def get_embeddings(self, input_ids, attention_mask, pixel_values):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
        return outputs.image_embeds, outputs.text_embeds
    
    def compute_modality_score(self, train_dataloader, 
                            baseline_text="[PAD]", baseline_image=None, mode='default'):
        """
        Compute modality scores for concepts in a concept bottleneck model.
        When mode='concat', don't check the multimodal concepts but the combined ones
        """


        if(self.combine_type=='text' or self.combine_type=='image'):
            raise ValueError("This function is only applicable for multimodal models.")

        self.eval()

        all_concept_logits_img = []
        all_concept_logits_txt = []
        all_concept_logits = []

        for batch in train_dataloader:

            labels = batch['label']
            images = batch['image'].to(self.device)
            texts = batch['text']

            inputs = self.processor(text=texts, return_tensors="pt", 
                                padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                concept_logits, concept_repr, concept_repr_combined, _ = self(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        pixel_values=images # raw images
                        ) # shape = (batch_size, num_concepts)
            
            if(mode!='concat'):
                if((self.concept_representation=='logits') or (self.concept_representation=='importance')):
                    # apply sigmoid in that case
                    all_concept_logits.append(torch.sigmoid(concept_repr.detach().cpu()))
                else:
                    # keep concept repr as they are
                    all_concept_logits.append(concept_repr.detach().cpu())
            else:
                if((self.concept_representation=='logits') or (self.concept_representation=='importance')):
                    all_concept_logits.append(torch.sigmoid(concept_repr_combined.detach().cpu()))
                else:
                    all_concept_logits.append(concept_repr_combined.detach().cpu())

            # Prepare a baseline image (all black) if not provided
            if baseline_image is None:
                baseline_image = torch.zeros_like(images[0]).unsqueeze(0).to(self.device)

            # 1. Image activations: real images + baseline text
            baseline_texts = [baseline_text] * len(images)
            inputs_img = self.processor(text=baseline_texts, return_tensors="pt", 
                                padding=True, truncation=True).to(self.device)
        
            with torch.no_grad():
                concept_logits_img, concept_repr_img, concept_repr_combined_img, _ = self(
                        input_ids=inputs_img.input_ids,
                        attention_mask=inputs_img.attention_mask,
                        pixel_values=images # raw images
                        ) # shape = (batch_size, num_concepts)

            if(mode!='concat'):
                if((self.concept_representation=='logits') or (self.concept_representation=='importance')):
                    # apply sigmoid in that case
                    all_concept_logits_img.append(torch.sigmoid(concept_repr_img.detach().cpu()))
                else:
                    # keep concept repr as they are
                    all_concept_logits_img.append(concept_repr_img.detach().cpu())
            else:
                if((self.concept_representation=='logits') or (self.concept_representation=='importance')):
                    all_concept_logits_img.append(torch.sigmoid(concept_repr_combined_img.detach().cpu()))
                else:   
                    all_concept_logits_img.append(concept_repr_combined_img.detach().cpu())
            
            # 2. Text activations: real texts + baseline image
            baseline_images = baseline_image.repeat(len(texts), 1, 1, 1)
            inputs_txt = self.processor(text=texts, return_tensors="pt", 
                                padding=True, truncation=True).to(self.device)
            
            with torch.no_grad():
                concept_logits_txt, concept_repr_txt, concept_repr_combined_txt, _ = self(
                        input_ids=inputs_txt.input_ids,
                        attention_mask=inputs_txt.attention_mask,
                        pixel_values=baseline_images
                        ) # shape = (batch_size, num_concepts)

            if(mode!='concat'):
                #all_concept_logits_txt.append(torch.sigmoid(concept_logits_txt).cpu())
                if((self.concept_representation=='logits') or (self.concept_representation=='importance')):
                    # apply sigmoid in that case
                    all_concept_logits_txt.append(torch.sigmoid(concept_repr_txt.detach().cpu()))
                else:
                    # keep concept repr as they are
                    all_concept_logits_txt.append(concept_repr_txt.detach().cpu())
            else:
                if((self.concept_representation=='logits') or (self.concept_representation=='importance')):
                    all_concept_logits_txt.append(torch.sigmoid(concept_repr_combined_txt.detach().cpu()))
                else:
                    all_concept_logits_txt.append(concept_repr_combined_txt.detach().cpu())

        # 3. Compute mean activations for each concept
        E_img = torch.cat(all_concept_logits_img, dim=0).mean(dim=0) # shape = (num_concepts,)
        E_txt = torch.cat(all_concept_logits_txt, dim=0).mean(dim=0)

        if(True):
            # just check activations when concept is actually activated with full model
            all_concept_logits_img = torch.cat(all_concept_logits_img, dim=0)
            all_concept_logits_txt = torch.cat(all_concept_logits_txt, dim=0)
            all_concept_logits = torch.cat(all_concept_logits, dim=0)

            print(all_concept_logits.shape)
            print(all_concept_logits_img.shape)
            print(all_concept_logits_txt.shape)

            # Mask for activations above threshold
            mask = all_concept_logits > 0.5  # Boolean, same shape

            # Sum activations only where activated
            sum_img = (all_concept_logits_img * mask).sum(dim=0)
            sum_txt = (all_concept_logits_txt * mask).sum(dim=0)

            print(sum_img)
            print(sum_img.shape)
            print(sum_txt)
            print(sum_txt.shape)

            # Count number of activations > threshold per concept
            count = mask.sum(dim=0).clamp(min=1)  # Avoid zero division

            print(count)
            print(count.shape)

            # Mean activation per concept, only over activated samples
            E_img = sum_img / count
            E_txt = sum_txt / count

            print(E_img)
            print(E_img.shape)
            print(E_txt)
            print(E_txt.shape)

        # 4. Compute modality scores
        modality_scores = E_img / (E_img + E_txt + 1e-8)

        return modality_scores.cpu().numpy()

    def sequential_train_model(self, train_dataloader, val_dataloader, test_dataloader, num_epochs=30, learning_rate=1e-5, fixed_lr=False, patience=5, max_len=512):
        # train backbone+concepts
        history_concepts, best_model_state = self.train_model(train_dataloader, val_dataloader, test_dataloader, num_epochs=num_epochs, learning_rate=learning_rate, fixed_lr=fixed_lr, patience=patience, max_len=max_len, sequential='concepts')
        # load model with best concept acc
        self.load_state_dict(best_model_state)
        # train final classifier only
        history_targets, best_model_state = self.train_model(train_dataloader, val_dataloader, test_dataloader, num_epochs=num_epochs, learning_rate=learning_rate, fixed_lr=fixed_lr, patience=patience, max_len=max_len, sequential='targets')
        return history_targets, best_model_state
    
    def independant_train_model(self, train_dataloader, val_dataloader, test_dataloader, num_epochs=30, learning_rate=1e-5, fixed_lr=False, patience=5, max_len=512):
        # train backbone+concepts
        history_concepts, best_model_state = self.train_model(train_dataloader, val_dataloader, test_dataloader, num_epochs=num_epochs, learning_rate=learning_rate, fixed_lr=fixed_lr, patience=patience, max_len=max_len, sequential='concepts')
        # load model with best concept acc
        self.load_state_dict(best_model_state)
        # train final classifier only
        history_targets, best_model_state = self.train_model(train_dataloader, val_dataloader, test_dataloader, num_epochs=num_epochs, learning_rate=learning_rate, fixed_lr=fixed_lr,  patience=patience, max_len=max_len, sequential='targets', independant=True)
        return history_targets, best_model_state

    def train_model(self, train_dataloader, val_dataloader, test_dataloader, num_epochs=30, learning_rate=1e-5, fixed_lr=False, patience=5, max_len=512, sequential=False, independant=False):
        self.to(self.device)

        best_val_accuracy = 0
        best_val_f1 = 0
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        alpha = 0.05 # smoothing factor to calculate running average loss

        # calculate pos_weight on train_dataset, this improves detection when True/False are unbalanced for each concept
        # only used when concepts are binary

        if(self.pos_weight):
            print('pos_weight = True')
            
            all_train_concept_labels = []

            with torch.no_grad():
                for batch in train_dataloader:
                    concept_labels = torch.stack(
                        [batch[col] for col in self.concept_list], dim=1
                    ).float()  # [batch_size, n_concepts]
                    all_train_concept_labels.append(concept_labels.cpu())  # keep as tensor, save memory

            # Concatenate at the end for a big tensor [num_samples, n_concepts]
            all_train_concept_labels = torch.cat(all_train_concept_labels, dim=0)

            pos_ratio = all_train_concept_labels.mean(dim=0)
            pos_weight = (1.0 - pos_ratio) / (pos_ratio + 1e-8)
            pos_weight = pos_weight / pos_weight.mean()
        
            criterion_concept = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(self.device))

        else:

            criterion_concept = nn.BCEWithLogitsLoss()

        # different loss if concept activation are represented by importance scores
        if(self.concept_representation == 'importance'):
            if(self.loss_CBLLM=='MSE'):
                criterion_concept = nn.MSELoss()  # or nn.L1Loss() # needs flattened inputs & target
            elif(self.loss_CBLLM=='cos_cubed'):
                criterion_concept = lambda logits,labels: 1.-cos_sim_cubed(logits, labels)  # # [0, 2] since cos_sim_cubed is [-1, 1], minimize to 0 to maximize similarity metric [closer to 1]
            else:
                raise ValueError('loss_CBLLM not implemented')

        # task loss     
        criterion = nn.CrossEntropyLoss()

        # Default: unfreeze all
        for param in self.parameters():
            param.requires_grad = True

        # unfreeze the last layer only
        if(self.frozen_backbone=='not_last_layer'):
            for name, param in self.base_model.named_parameters():
                if ("visual_proj" in name) or ("text_proj" in name):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        if(self.combine_type=='image'):
            # freeze text part of clip
            for name, param in self.base_model.named_parameters():
                if "text_model" in name:
                    param.requires_grad = False
        elif(self.combine_type=='text'):
            # freeze image part of clip
            for name, param in self.base_model.named_parameters():
                if "visual_model" in name:
                    param.requires_grad = False
        
        # unfreeze all clip (text & image)
        if(self.contrastive):
            for name, param in self.base_model.named_parameters():
                param.requires_grad = True

        if(sequential=="concepts"):
            # Freeze classifier, unfreeze others
            for name, param in self.named_parameters():
                if name.startswith("classifier"):
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        elif(sequential=="targets"):
            # Unfreeze classifier, freeze others
            for name, param in self.named_parameters():
                if name.startswith("classifier"):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        # freeze concept_layer parameters because not used in predictions
        if(self.contrastive):
            if(self.combine_type!='concat'):
                for name, param in self.concept_layer.named_parameters():
                    param.requires_grad = False
            else:
                for name, param in self.concept_layer_text.named_parameters():
                    param.requires_grad = False
                for name, param in self.concept_layer_image.named_parameters():
                    param.requires_grad = False
        

        if(fixed_lr):
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=learning_rate)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs-1, eta_min=learning_rate)
        else:
            # learning rate for each component of the model
            param_groups = []
            # Add CLIP parameters if any require gradients
            clip_params = [p for p in self.base_model.parameters() if p.requires_grad]
            if len(clip_params)>0:
                param_groups.append({'params': clip_params, 'lr': learning_rate})
            # Add concept layer parameters if any require gradients
            if(self.combine_type!='concat'):
                concept_params = [p for p in self.concept_layer.parameters() if p.requires_grad]
            else:
                concept_params = [p for p in self.concept_layer_text.parameters() if p.requires_grad] + [p for p in self.concept_layer_image.parameters() if p.requires_grad]
            if concept_params:
                param_groups.append({'params': concept_params, 'lr': 1e-1})
            # Add classifier parameters if any require gradients
            classifier_params = [p for p in self.classifier.parameters() if p.requires_grad]
            if classifier_params:
                param_groups.append({'params': classifier_params, 'lr': 1e-1})

            optimizer = torch.optim.AdamW(param_groups)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs-1, eta_min=1e-5)
        
        # display list of parameters that have requires_grad=True
        #for name, param in self.named_parameters():
        #    if param.requires_grad:
        #        print(name)

        # Add concept metrics to history
        history = {
                'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
                'train_f1': [], 'val_f1': [],
                'train_class_f1': [], 'val_class_f1': [],
                'train_concept_acc': [], 'val_concept_acc': [],
                'train_concept_f1': [], 'val_concept_f1': [],
                'train_per_concept_acc': [], 'val_per_concept_acc': [],
                'train_per_concept_f1': [], 'val_per_concept_f1': [],
                'test_loss': [],'test_acc': [],
                'test_f1': [],'test_class_f1': [],
                'test_concept_acc': [], "test_per_concept_acc":[],'test_concept_f1': [],'test_per_concept_f1': [],
                'train_concept_leakage': [], 'val_concept_leakage': [], 'test_concept_leakage': [],
                'train_concept_interleakage': [], 'val_concept_interleakage': [], 'test_concept_interleakage': [],
                'train_per_concept_leakage': [], 'val_per_concept_leakage': [], 'test_per_concept_leakage': [],
                'train_per_concept_interleakage': [], 'val_per_concept_interleakage': [], 'test_per_concept_interleakage': []
                }

        running_mean_final_train_loss = 0.0
        running_mean_concept_train_loss = 0.0
        running_mean_leakage_concept_train_loss = 0.0

        for epoch in range(num_epochs):
            self.train()
            train_losses = []
            #final_train_losses = []
            #concept_train_losses = []
            #leakage_concept_train_losses = []
            all_train_predictions = []
            all_train_labels = []
            
            # Track concept predictions and labels
            all_train_concept_preds = []
            all_train_concept_repr = []
            all_train_concept_labels = []
            all_train_concept_logits = []

            train_loop = tqdm(enumerate(train_dataloader), desc=f"Epoch {epoch+1}/{num_epochs} [Train] (lr={[group['lr'] for group in optimizer.param_groups]})")
            for batch_idx, batch in train_loop:

                labels = batch['label']
                if((self.combine_type=='concat') or (self.combine_type=='combine')):
                    texts = batch['text']
                    images = batch['image']
                elif(self.combine_type=='text'):
                    texts = batch['text']
                    images = torch.zeros(len(texts), 3, 224, 224, dtype=torch.float).to(self.device)
                elif(self.combine_type=='image'):
                    images = batch['image']
                    texts = [""] * len(images)

                label_indices = torch.tensor([label for label in labels]).to(self.device)
                images = images.to(self.device)
                inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)

                optimizer.zero_grad()
                # temperature annealing for concept learning, inspired from https://arxiv.org/pdf/1611.01144
                tau = max(0.5, np.exp(-(-np.log(0.5)*2/num_epochs)*epoch))
                concept_logits, concept_repr, concept_repr_combined, logits = self(input_ids=inputs.input_ids, 
                                                    attention_mask=inputs.attention_mask, 
                                                    pixel_values=images, # use raw images because already transformed in dataloader
                                                    tau=tau)

                if(independant):
                    # only use ground truth concept to predict final class
                    if(self.combine_type!='concat'):
                        concept_labels = torch.stack([batch[col] for col in self.concept_list], dim=1).to(self.device).to(dtype=torch.float32) # Shape: [batch_size, n_concepts]
                        logits = self.classifier(concept_labels)
                    else:
                        # extract concept_labels for text and image concepts
                        concept_labels_texts = torch.stack([batch[col] for col in self.concept_list if 'concept_text_' in col], dim=1).to(self.device).to(dtype=torch.float32)
                        concept_labels_images = torch.stack([batch[col] for col in self.concept_list if 'concept_image_' in col], dim=1).to(self.device).to(dtype=torch.float32)
                        # Apply 'OR' operation to logits in order to combine them
                        concept_logits, concept_repr, concept_repr_combined = self._or_concept_layer(text_logits=concept_labels_texts, image_logits=concept_labels_images, tau=tau)
                        logits = self.classifier(concept_repr_combined)

                # elastic net of final classifier
                #l1_norm = torch.norm(self.classifier.weight, p=1)
                #l2_norm = torch.norm(self.classifier.weight, p=2)

                # combine concept & task losses
                loss_final_task = criterion(logits, label_indices) #+ self.alpha * (self.l1_ratio * l1_norm + (1 - self.l1_ratio) * l2_norm)
                if(sequential=='concepts'):
                    # dummy loss final task when training for concepts
                    loss_final_task = torch.tensor(1.).to(self.device)
                #final_train_losses += [loss_final_task.item()]
                running_mean_final_train_loss = (1 - alpha)*running_mean_final_train_loss + alpha*loss_final_task.item() #(running_mean_final_train_loss * batch_idx + loss_final_task.item()) / (batch_idx + 1)

                # get concept labels & loss
                if (self.concept_training):
                    concept_labels = torch.stack([batch[col] for col in self.concept_list], dim=1).to(self.device).float()  # Shape: [batch_size, n_concepts]
                    
                    # Get binary concept predictions for metrics regardless of representation mode
                    #print('type concept logits')
                    #print(type(concept_logits))
                    concept_binary = self._get_binary_concepts(concept_logits, tau=tau)
                    
                    # Store original concept labels and predictions for metrics calculation
                    all_train_concept_labels.extend(concept_labels.detach().cpu().numpy()) # Shape: [n_batch, n_concepts]
                    all_train_concept_preds.extend(concept_binary.detach().cpu().numpy())
                    all_train_concept_repr.extend(concept_repr.detach().cpu().numpy())
                    all_train_concept_logits.extend(concept_logits.detach().cpu().numpy())
                    
                    #concept_labels_flat = torch.t(concept_labels).contiguous().view(-1).float()  # Shape: [n_concepts*batch_size]
                    #concept_logits_flat = torch.t(concept_logits).contiguous().view(-1).float()  # Shape: [n_concepts*batch_size]

                    if((self.cpo==False) or (self.concept_representation == 'importance')):
                        loss_concept = criterion_concept(concept_logits, concept_labels)
                    elif(self.cpo==True):
                        loss_concept = cpo_loss(concept_logits, concept_labels, temperature=tau)
                    #concept_train_losses += [loss_concept.item()]
                    running_mean_concept_train_loss = (1 - alpha)*running_mean_concept_train_loss + alpha*loss_concept.item() # (running_mean_concept_train_loss * batch_idx + loss_concept.item()) / (batch_idx + 1)
                else:
                    loss_concept = torch.zeros(1, device=self.device)

                # calculating leakage loss
                if(self.leakage_loss):
                    train_per_concept_leakage = {}
                    loss_leakage_concept = 0.
                    for i, concept_name in enumerate(self.concept_list):
                        train_per_concept_leakage[concept_name] = pytorch_concept_task_leakage_differentiable(concept_repr[: ,i], concept_labels[:, i], label_indices, target_continuous=True if self.concept_representation=='importance' else False)
                        #if(epoch==0):
                        loss_leakage_concept += train_per_concept_leakage[concept_name]/len(self.concept_list)
                        #else:
                        #    if(self.concept_representation == 'importance'):
                        #        loss_leakage_concept += train_per_concept_leakage[concept_name]/len(self.concept_list)*train_per_concept_acc[concept_name] # weight by RMSE
                        #    else:
                        #        loss_leakage_concept += train_per_concept_leakage[concept_name]/len(self.concept_list)*train_per_concept_f1[concept_name] # weight by f1 score

                    #leakage_concept_train_losses += [loss_leakage_concept.item()]
                    running_mean_leakage_concept_train_loss = (1 - alpha)*running_mean_leakage_concept_train_loss + alpha*loss_leakage_concept.item() # (running_mean_leakage_concept_train_loss * batch_idx + loss_leakage_concept.item()) / (batch_idx + 1)


                # calculate mean losses until now
                #mean_loss = np.mean(final_train_losses)
                #mean_concept_loss = np.mean(concept_train_losses)
                #mean_leakage_concept_loss = np.mean(leakage_concept_train_losses)

                # scaling factor for concept loss
                if(self.loss_rescaled):
                    loss_concept *= self.lambda_XtoC/(running_mean_concept_train_loss+1e-8)*running_mean_final_train_loss
                else:
                    loss_concept *= self.lambda_XtoC

                if(sequential=="concepts"):
                    loss = loss_concept
                elif(sequential=="targets"):
                    loss = loss_final_task
                else:
                    loss = loss_concept + loss_final_task

                # add leakage loss to overall loss
                if(self.leakage_loss and sequential!="targets"):
                    # activation_for_leakage_loss = eta_min + 0.5*(initial_lr-eta_min)*(1+math.cos(math.pi*step/T_max))

                    if(self.leakage_loss_activation=='up'):
                        activation_for_leakage_loss = 1 + 0.5*(0-1)*(1+math.cos(math.pi*epoch/(num_epochs-1))) # cosine decay, from 0 to 1, to smoothly activate leakage loss over epochs
                    elif(self.leakage_loss_activation=='down'):
                        activation_for_leakage_loss = 0 + 0.5*(1-0)*(1+math.cos(math.pi*epoch/(num_epochs-1))) # cosine decay, from 1 to 0, to smoothly deactivate leakage loss over epochs
                    else:
                        activation_for_leakage_loss = self.leakage_loss_activation
                    
                    if(self.loss_rescaled):
                        loss_leakage_concept *= activation_for_leakage_loss*self.lambda_XtoC_leakage/(running_mean_leakage_concept_train_loss+1e-8)*running_mean_final_train_loss
                    else:
                        loss_leakage_concept *= activation_for_leakage_loss*self.lambda_XtoC_leakage

                    #print(loss_leakage_concept.requires_grad)  # Should be True

                    loss += loss_leakage_concept

                loss.backward()
                optimizer.step()

                probabilities = torch.nn.functional.softmax(logits, dim=1)
                predictions = logits.argmax(dim=1)

                all_train_predictions.extend(predictions.cpu().numpy())
                all_train_labels.extend(label_indices.cpu().numpy())
                train_losses += [loss.item()]
                if(not self.leakage_loss):
                    train_loop.set_postfix(total_loss=loss.item(), loss=loss_final_task.item(), concept_loss=loss_concept.item())
                else:
                    train_loop.set_postfix(total_loss=loss.item(), loss=loss_final_task.item(), concept_loss=loss_concept.item(), leakage_concept_loss=loss_leakage_concept.item())

            scheduler.step()

            # Calculate training metrics for task
            train_loss = np.sum(train_losses)/len(train_dataloader)
            all_train_predictions = np.array(all_train_predictions)
            all_train_labels = np.array(all_train_labels)
            train_accuracy = accuracy_score(all_train_labels.flatten(), all_train_predictions.flatten())
            train_f1 = f1_score(all_train_labels, all_train_predictions, average='macro')

            # Per-class metrics
            train_class_f1 = dict(zip(self.class_dict.values(),f1_score(all_train_labels, all_train_predictions, average=None)))

            # Initialize per-concept metrics
            train_concept_acc = 0.0
            train_concept_f1 = 0.0
            concept_f1_score = 0.0
            train_per_concept_acc = {}
            train_per_concept_f1 = {}
            train_per_concept_leakage = {}
            train_per_concept_interleakage = {}
            
            # Calculate concept metrics if binary concept training is enabled
            if (self.concept_training):
                # Convert to numpy arrays for metric calculation
                all_train_concept_preds = np.array(all_train_concept_preds)
                all_train_concept_labels = np.array(all_train_concept_labels)
                all_train_concept_repr = np.array(all_train_concept_repr)
                all_train_concept_logits = np.array(all_train_concept_logits)
                
                # Calculate concept accuracy and F1 score
                if(self.concept_representation!='importance'):
                    train_concept_acc = accuracy_score(all_train_concept_labels.flatten(), all_train_concept_preds.flatten())
                    train_concept_f1 = f1_score(all_train_concept_labels, all_train_concept_preds, average='macro')
                else:
                    # need a metric for concept acc/f1 CBLLM
                    # Compute R² across all concepts (higher is better)
                    #train_concept_acc = r2_score(all_train_concept_labels.flatten(), all_train_concept_logits.flatten())
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            category=FutureWarning,
                            message=".*root_mean_squared_error.*"
                            )
                        train_concept_acc = mean_squared_error(all_train_concept_labels.flatten(), all_train_concept_logits.flatten(),squared=False)

                    # Compute cosine similarity across all concepts (higher is better)
                    train_concept_f1 = np.dot(all_train_concept_labels.flatten(), all_train_concept_logits.flatten()) / (np.linalg.norm(all_train_concept_labels.flatten()) * np.linalg.norm(all_train_concept_logits.flatten()) + 1e-8)

                concept_train_leakage = 0.
                concept_train_interleakage = 0.
                n_interleakage = 0
                # Calculate per-concept F1 scores
                for i, concept_name in enumerate(self.concept_list):
                    concept_true = all_train_concept_labels[:, i]
                    concept_pred = all_train_concept_preds[:, i]
                    concept_repr = all_train_concept_repr[: ,i]
                    concept_logits = all_train_concept_logits[:, i]

                    if(self.concept_representation!='importance'):
                        concept_acc = accuracy_score(concept_true, concept_pred)
                        train_per_concept_acc[concept_name] = concept_acc
                        concept_f1_score = f1_score(concept_true, concept_pred, average='binary')
                        train_per_concept_f1[concept_name] = concept_f1_score
                    else:
                        # R² score for regression performance (higher is better, 1.0 is perfect)
                        #train_per_concept_acc[concept_name] = r2_score(concept_true, concept_logits)
                        with warnings.catch_warnings():
                            warnings.filterwarnings(
                            "ignore",
                            category=FutureWarning,
                            message=".*root_mean_squared_error.*"
                            )
                            train_per_concept_acc[concept_name] = mean_squared_error(concept_true, concept_logits,squared=False)
                        
                        # Cosine similarity (higher is better, 1.0 is perfect alignment)
                        # Need to reshape for sklearn's cosine_similarity
                        train_per_concept_f1[concept_name] = np.dot(concept_true, concept_logits) / (np.linalg.norm(concept_true) * np.linalg.norm(concept_logits) + 1e-8)

                    train_per_concept_leakage[concept_name] = concept_task_leakage(concept_repr, concept_true, all_train_labels, target_continuous=True if self.concept_representation=='importance' else False)
                    concept_train_leakage += train_per_concept_leakage[concept_name]/len(self.concept_list)

                    # calculer inter-concept scores
                    for j, target_concept in enumerate(self.concept_list):
                        if i == j:
                            continue

                        n_interleakage += 1

                        target_repr = all_train_concept_repr[:, j]  # représentation du concept target
                        target_true = all_train_concept_labels[:, j]

                        leakage_score = interconcept_leakage(concept_repr, target_repr, concept_true, target_true, target_continuous=True if self.concept_representation=='importance' else False)
                        train_per_concept_interleakage[f"{concept_name} -> {target_concept}"] = leakage_score

                        concept_train_interleakage += train_per_concept_interleakage[f"{concept_name} -> {target_concept}"]


                print('train_concept_f1: ',train_concept_f1)
                # now calculte of all concept f1 scores
                print('mean of train_per_concept_f1: ',np.mean(list(train_per_concept_f1.values())))
                # same with accuracy
                print('train_concept_acc: ',train_concept_acc)
                print('mean of train_per_concept_acc: ',np.mean(list(train_per_concept_acc.values())))


                # mean interleakage
                concept_train_interleakage /= n_interleakage

                # Average concept loss
                concept_train_loss = running_mean_concept_train_loss # np.sum(concept_train_losses)/len(train_dataloader)



            # Record metrics
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_accuracy)
            history['train_f1'].append(train_f1)
            history['train_class_f1'].append(train_class_f1)

            # Validation phase
            #val_all_concept_preds, val_loss, val_accuracy, val_f1, val_class_f1, val_concept_acc, val_concept_f1, val_per_concept_acc, val_per_concept_f1, concept_val_leakage, val_per_concept_leakage, val_interconcept_leakage = self.evaluate(val_dataloader, sequential=sequential, plot_concepts=True if ((epoch==0) or (epoch==(num_epochs-1))) and (self.num_classes<5) else False)

            # put all val variables to 0 instead
            val_all_concept_preds, val_loss, val_accuracy, val_f1, val_class_f1, val_concept_acc, val_concept_f1, val_per_concept_acc, val_per_concept_f1, concept_val_leakage, val_per_concept_leakage, val_interconcept_leakage = 0., 0., 0., 0., {class_name:0 for class_name in self.class_dict.values()}, 0., 0., {concept_name:0 for concept_name in self.concept_list}, {concept_name:0 for concept_name in self.concept_list}, 0., {concept_name:0 for concept_name in self.concept_list}, {concept_name:0 for concept_name in self.concept_list}

            # Record metrics
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_accuracy)
            history['val_f1'].append(val_f1)
            history['val_class_f1'].append(val_class_f1)

            # Record concept metrics
            if (self.concept_training):
                history['train_concept_acc'].append(train_concept_acc)
                history['train_concept_f1'].append(train_concept_f1)
                history['train_per_concept_acc'].append(train_per_concept_acc)
                history['train_per_concept_f1'].append(train_per_concept_f1)
                
                history['val_concept_acc'].append(val_concept_acc)
                history['val_concept_f1'].append(val_concept_f1)
                history['val_per_concept_acc'].append(val_per_concept_acc)
                history['val_per_concept_f1'].append(val_per_concept_f1)

                history['train_concept_leakage'].append(concept_train_leakage)
                history['train_concept_interleakage'].append(concept_train_interleakage)
                history['train_per_concept_leakage'].append(train_per_concept_leakage)
                history['train_per_concept_interleakage'].append(train_per_concept_interleakage)
                history['val_concept_leakage'].append(concept_val_leakage)
                history['val_concept_interleakage'].append(np.mean(list(val_interconcept_leakage.values())))
                history['val_per_concept_leakage'].append(concept_val_leakage)
                history['val_per_concept_interleakage'].append(val_interconcept_leakage)
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Train F1: {train_f1:.4f}")
            
            # Print concept metrics if concept training is enabled
            if (self.concept_training):
                if(self.concept_representation=='importance'):
                    print(f"Train Concept Loss: {concept_train_loss:.4f}, Train Concept RMSE: {train_concept_acc:.4f}, Train Concept Cosine Sim: {train_concept_f1:.4f}, Train Concept Leakage: {concept_train_leakage:.4f}, Train Concept Interleakage: {concept_train_interleakage:.4f}")
                else:
                    print(f"Train Concept Loss: {concept_train_loss:.4f}, Train Concept Acc: {train_concept_acc:.4f}, Train Concept F1: {train_concept_f1:.4f}, Train Concept Leakage: {concept_train_leakage:.4f}, Train Concept Interleakage: {concept_train_interleakage:.4f}")
            
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val F1: {val_f1:.4f}")
            
            # Print validation concept metrics
            if (self.concept_training):
                if(self.concept_representation=='importance'):
                    print(f"Val Concept RMSE: {float(val_concept_acc):.4f}, Val Concept Cosine Sim: {float(val_concept_f1):.4f}, Val Concept Leakage: {float(concept_val_leakage):.4f}, Val Concept Interleakage: {float(np.mean(list(val_interconcept_leakage.values()))):.4f}")
                else:
                    print(f"Val Concept Acc: {float(val_concept_acc):.4f}, Val Concept F1: {float(val_concept_f1):.4f}, Val Concept Leakage: {float(concept_val_leakage):.4f}, Val Concept Interleakage: {float(np.mean(list(val_interconcept_leakage.values()))):.4f}")

            # save model with best val_accuracy
            # Early stopping check
            # if(sequential=='concepts'):
            #     if(self.concept_representation!='importance'):
            #         # take concept_f1 as measure to save best model
            #         if(val_concept_f1 > best_val_f1):
            #             best_val_f1 = val_concept_f1
            #             best_model_state = copy.deepcopy(self.state_dict())
            #             epochs_without_improvement = 0
            #         else:
            #             epochs_without_improvement += 1
            #     else:
            #         # take val_loss (which corresponds to concept loss) as measure to save best model
            #         if(val_loss < best_val_loss):
            #             best_val_loss = val_loss
            #             best_model_state = copy.deepcopy(self.state_dict())
            #             epochs_without_improvement = 0
            #         else:
            #             epochs_without_improvement += 1
            # else:
            #     if(val_f1 > best_val_f1):
            #         best_val_f1 = val_f1
            #         best_model_state = copy.deepcopy(self.state_dict())
            #         epochs_without_improvement = 0
            #     else:
            #         epochs_without_improvement += 1

            best_model_state = copy.deepcopy(self.state_dict())
            best_epoch = epoch

            # if epochs_without_improvement >= patience:
            #     print(f"Early stopping triggered after {epoch+1} epochs")
            #     break
            
        print('Computing test metrics...')
        # load best model state
        self.load_state_dict(best_model_state)
        # now calculate test metrics with evaluate
        test_all_concept_preds, test_loss, test_accuracy, test_f1, test_class_f1, test_concept_acc, test_concept_f1, test_per_concept_acc, test_per_concept_f1, concept_test_leakage, test_per_concept_leakage, test_interconcept_leakage = self.evaluate(test_dataloader, sequential=sequential)
        
        # add test metrics to history
        history['test_loss'] = test_loss
        history['test_acc'] = test_accuracy
        history['test_f1'] = test_f1
        history['test_class_f1'] = test_class_f1
        if (self.concept_training):
            history['test_concept_acc'] = test_concept_acc
            history['test_concept_f1'] = test_concept_f1
            history['test_per_concept_acc'] = test_per_concept_acc
            history['test_per_concept_f1'] = test_per_concept_f1
            history['test_concept_leakage'] = concept_test_leakage
            history['test_concept_interleakage'] = np.mean(list(test_interconcept_leakage.values()))
            history['test_per_concept_leakage'] = test_per_concept_leakage
            history['test_per_concept_interleakage'] = test_interconcept_leakage

        # print best performances associated with best_val_accuracy
        # if(sequential=='concepts'):
        #     if(self.concept_representation!='importance'):
        #         #best_epoch = int(np.argmax(history['val_concept_f1']))
        #         print(f"\n=== Best Epoch: {best_epoch+1} (val_f1 = {history['val_concept_f1'][best_epoch]:.4f}) ===")
        #     else:
        #         #best_epoch = int(np.argmax(history['val_loss']))
        #         print(f"\n=== Best Epoch: {best_epoch+1} (val_loss = {history['val_loss'][best_epoch]:.4f}) ===")
        # else:
        #     #best_epoch = int(np.argmax(history['val_f1']))
        #     print(f"\n=== Best Epoch: {best_epoch+1} (val_f1 = {history['val_f1'][best_epoch]:.4f}) ===")

        print(f"\n=== Final Results for Epoch: {best_epoch+1} ===")

        # Display metrics
        print(f"Train Acc: {history['train_acc'][best_epoch]:.4f}, Val Acc: {history['val_acc'][best_epoch]:.4f}, Test Acc: {history['test_acc']:.4f}")
        print(f"Train F1: {history['train_f1'][best_epoch]:.4f}, Val F1: {history['val_f1'][best_epoch]:.4f}, Test F1: {history['test_f1']:.4f} ")
        if (self.concept_training):
            if(self.concept_representation=='importance'):
                print(f"Train Concept RMSE: {float(history['train_concept_acc'][best_epoch]):.4f}, "
                    f"Val Concept RMSE: {float(history['val_concept_acc'][best_epoch]):.4f}, "
                    f"Test Concept RMSE: {float(history['test_concept_acc']):.4f}")
                print(f"Train Concept Cosine Sim: {float(history['train_concept_f1'][best_epoch]):.4f}, "
                    f"Val Concept Cosine Sim: {float(history['val_concept_f1'][best_epoch]):.4f}, "
                    f"Test Concept Cosine Sim: {float(history['test_concept_f1']):.4f}")
                print(f"Train Concept Leakage: {float(history['train_concept_leakage'][best_epoch]):.4f}, "
                    f"Val Concept Leakage: {float(history['val_concept_leakage'][best_epoch]):.4f}, "
                    f"Test Concept Leakage: {float(history['test_concept_leakage']):.4f}")
                print(f"Train Concept Interleakage: {float(history['train_concept_interleakage'][best_epoch]):.4f}, "
                    f"Val Concept Interleakage: {float(history['val_concept_interleakage'][best_epoch]):.4f}, "
                    f"Test Concept Interleakage: {float(history['test_concept_interleakage']):.4f}")
            else:
                print(f"Train Concept Acc: {float(history['train_concept_acc'][best_epoch]):.4f}, "
                    f"Val Concept Acc: {float(history['val_concept_acc'][best_epoch]):.4f}, "
                    f"Test Concept Acc: {float(history['test_concept_acc']):.4f}")
                print(f"Train Concept F1: {float(history['train_concept_f1'][best_epoch]):.4f}, "
                    f"Val Concept F1: {float(history['val_concept_f1'][best_epoch]):.4f}, "
                    f"Test Concept F1: {float(history['test_concept_f1']):.4f}")
                print(f"Train Concept Leakage: {float(history['train_concept_leakage'][best_epoch]):.4f}, "
                    f"Val Concept Leakage: {float(history['val_concept_leakage'][best_epoch]):.4f}, "
                    f"Test Concept Leakage: {float(history['test_concept_leakage']):.4f}")
                print(f"Train Concept Interleakage: {float(history['train_concept_interleakage'][best_epoch]):.4f}, "
                    f"Val Concept Interleakage: {float(history['val_concept_interleakage'][best_epoch]):.4f}, "
                    f"Test Concept Interleakage: {float(history['test_concept_interleakage']):.4f}")
            
        return history, best_model_state


    def evaluate(self, dataloader, sequential=False, plot_concepts=False):
        """
        Evaluate the model on the given dataloader
        
        Returns:
            float: Average loss
            float: Accuracy
            float: F1 score (macro)
            dict: Per-class F1 scores
            float: Concept accuracy (if concept_training is True)
            float: Concept F1 score (if concept_training is True)
            dict: Per-concept F1 scores (if concept_training is True)
        """
        self.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        # For concept evaluation
        all_concept_preds = []
        all_concept_repr = []
        all_concept_logits = []
        all_concept_labels = []

        random.seed(42)
        concept1 = 'concept_music festivals'
        concept2 = 'concept_fan culture'
        #concept3 = 'concept_ethical considerations'
        #concept4 = 'concept_ingredient features'
        if(concept1 not in self.concept_list):
            # randomly select a concept from concept_list
            concept1 = random.choice(self.concept_list)
        if(concept2 not in self.concept_list):
            # randomly select a concept from concept_list
            concept2 = random.choice(self.concept_list)
        all_concept_logits_concept1 = {classnumber: [] for classnumber in range(len(self.class_dict))}
        all_concept_logits_concept1.update({True: [], False: []})
        all_concept_logits_concept2 = {classnumber: [] for classnumber in range(len(self.class_dict))}
        all_concept_logits_concept2.update({True: [], False: []})
        #all_concept_logits_concept3 = {classnumber: [] for classnumber in range(len(self.class_dict))}
        #all_concept_logits_concept4 = {classnumber: [] for classnumber in range(len(self.class_dict))}

        with torch.no_grad():
            val_loop = tqdm(enumerate(dataloader), desc="Validation", leave=False)
            for batch_idx, batch in val_loop:

                labels = batch['label']
                if((self.combine_type=='concat') or (self.combine_type=='combine')):
                    texts = batch['text']
                    images = batch['image']
                elif(self.combine_type=='text'):
                    texts = batch['text']
                    images = torch.zeros(len(texts), 3, 224, 224, dtype=torch.float).to(self.device)
                elif(self.combine_type=='image'):
                    images = batch['image']
                    texts = [""] * len(images)

                label_indices = torch.tensor([label for label in labels]).to(self.device)
                images = images.to(self.device)
                inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)

                # Forward pass
                concept_logits, concept_repr, concept_repr_combined, logits = self(input_ids=inputs.input_ids, 
                                              attention_mask=inputs.attention_mask, 
                                              pixel_values=images) # use raw images because already transformed in dataloader

                # # find concept_logits values associated with a particular concept
                for classnumber in range(len(self.class_dict)):
                    all_concept_logits_concept1[classnumber] += (concept_logits[logits.argmax(dim=1)==classnumber, self.concept_list.index(concept1)]).tolist()
                    all_concept_logits_concept2[classnumber] += (concept_logits[logits.argmax(dim=1)==classnumber, self.concept_list.index(concept2)]).tolist()
                #     all_concept_logits_concept1[classnumber] += gumbel_sigmoid(concept_logits[logits.argmax(dim=1)==classnumber, self.concept_list.index(concept1)]).tolist()
                #     all_concept_logits_concept2[classnumber] += gumbel_sigmoid(concept_logits[logits.argmax(dim=1)==classnumber, self.concept_list.index(concept2)]).tolist()
                #     all_concept_logits_concept3[classnumber] += gumbel_sigmoid(concept_logits[logits.argmax(dim=1)==classnumber, self.concept_list.index(concept3)]).tolist()
                #     all_concept_logits_concept4[classnumber] += gumbel_sigmoid(concept_logits[logits.argmax(dim=1)==classnumber, self.concept_list.index(concept4)]).tolist()

                # Calculate concept loss if concept training is enabled
                if self.concept_training:
                        
                    concept_labels = torch.stack([batch[col] for col in self.concept_list], dim=1).to(self.device)
                    
                    # Get binary concept predictions for metrics
                    concept_binary = self._get_binary_concepts(concept_logits)

                    # record when concept is correctly predicted, logits > 0 and concept_binary=1, or when logits < 0 and concept_binary = 0
                    all_concept_logits_concept1[True] += (concept_logits[(concept_binary[:, self.concept_list.index(concept1)]==concept_labels[:, self.concept_list.index(concept1)]), self.concept_list.index(concept1)]).tolist()
                    all_concept_logits_concept1[False] += (concept_logits[(concept_binary[:, self.concept_list.index(concept1)]!=concept_labels[:, self.concept_list.index(concept1)]), self.concept_list.index(concept1)]).tolist()
                    all_concept_logits_concept2[True] += (concept_logits[(concept_binary[:, self.concept_list.index(concept2)]==concept_labels[:, self.concept_list.index(concept2)]), self.concept_list.index(concept2)]).tolist()
                    all_concept_logits_concept2[False] += (concept_logits[(concept_binary[:, self.concept_list.index(concept2)]!=concept_labels[:, self.concept_list.index(concept2)]), self.concept_list.index(concept2)]).tolist()
                    
                    # Store concept labels and binary predictions for metrics
                    all_concept_labels.extend(concept_labels.cpu().numpy())
                    all_concept_preds.extend(concept_binary.cpu().numpy())
                    all_concept_logits.extend(concept_logits.cpu().numpy())
                    all_concept_repr.extend(concept_repr.cpu().numpy())

                predictions = logits.argmax(dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(label_indices.cpu().numpy())

        # Calculate metrics
        avg_loss = 0.#total_loss / len(dataloader)
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        accuracy = accuracy_score(all_labels.flatten(), all_predictions.flatten())
        f1 = f1_score(all_labels, all_predictions, average='macro')

        # Per-class F1 scores
        class_f1 = dict(zip(self.class_dict.values(),f1_score(all_labels, all_predictions, average=None)))

        # Calculate concept metrics if concept training is enabled
        concept_acc, concept_f1 = 0.0, 0.0
        per_concept_f1 = {}
        per_concept_acc = {}
        per_concept_r2 = {}
        concept_val_leakage = 0
        concept_f1_score = 0.0
        per_concept_leakage = {}
        inter_concept_leakage = {}

        if (self.concept_training):
            # Convert to numpy arrays
            all_concept_preds = np.array(all_concept_preds)
            all_concept_labels = np.array(all_concept_labels)
            all_concept_repr = np.array(all_concept_repr)
            all_concept_logits = np.array(all_concept_logits)

            # Calculate overall metrics (all_concept_preds should already be binary)
            if(self.concept_representation!='importance'):
                concept_acc = accuracy_score(all_concept_labels.flatten(), all_concept_preds.flatten())
                concept_f1 = f1_score(all_concept_labels, all_concept_preds, average='macro')
            else:
                # need a metric for concept acc/f1 CBLLM
                # Continuous-valued concepts ("importance"): use regression-style metrics

                # Compute R² across all concepts (higher is better)
                #concept_acc = r2_score(all_concept_labels.flatten(), all_concept_logits.flatten())
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                            "ignore",
                            category=FutureWarning,
                            message=".*root_mean_squared_error.*"
                            )
                    concept_acc = mean_squared_error(all_concept_labels.flatten(), all_concept_logits.flatten(),squared=False)

                # Compute cosine similarity across all concepts (higher is better)
                concept_f1 = np.dot(all_concept_labels.flatten(), all_concept_logits.flatten()) / (np.linalg.norm(all_concept_labels.flatten()) * np.linalg.norm(all_concept_logits.flatten()) + 1e-8)

            # Calculate per-concept F1 scores
            for i, concept_name in enumerate(self.concept_list):
                concept_true = all_concept_labels[:, i]
                concept_pred = all_concept_preds[:, i]
                concept_repr = all_concept_repr[:, i]
                concept_logits = all_concept_logits[:, i]

                if(self.concept_representation!='importance'):
                    concept_f1_score = f1_score(concept_true, concept_pred, average='binary')
                    per_concept_f1[concept_name] = concept_f1_score
                    concept_acc = accuracy_score(concept_true, concept_pred)
                    per_concept_acc[concept_name] = concept_acc
                else:
                    # need a metric for concept acc/f1 CBLLM

                    # R² score for regression performance (higher is better, 1.0 is perfect)
                    #per_concept_acc[concept_name] = r2_score(concept_true, concept_logits)
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            category=FutureWarning,
                            message=".*root_mean_squared_error.*"
                            )
                        per_concept_acc[concept_name] = mean_squared_error(concept_true, concept_logits,squared=False)

                    per_concept_r2[concept_name] = r2_score(concept_true, concept_logits)
                    
                    # Cosine similarity (higher is better, 1.0 is perfect alignment)
                    # Need to reshape for sklearn's cosine_similarity
                    per_concept_f1[concept_name] = np.dot(concept_true, concept_logits) / (np.linalg.norm(concept_true) * np.linalg.norm(concept_logits) + 1e-8)

                per_concept_leakage[concept_name] = concept_task_leakage(concept_repr, concept_true, all_labels, target_continuous=True if self.concept_representation=='importance' else False)
                concept_val_leakage += per_concept_leakage[concept_name]/len(self.concept_list)

            # plot R² and rmse or cosine
            if(self.concept_representation=='importance'):

                rmse_vals = [per_concept_acc[c] for c in self.concept_list]
                r2_vals = [per_concept_r2[c] for c in self.concept_list]
                cos_vals = [per_concept_f1[c] for c in self.concept_list]

                fig, axes = plt.subplots(1, 2, figsize=(12, 5))

                # Left: R² as function of RMSE
                ax = axes[0]
                sc1 = ax.scatter(rmse_vals, r2_vals)
                for name, x, y in zip(self.concept_list, rmse_vals, r2_vals):
                    ax.annotate(name, (x, y), textcoords="offset points", xytext=(3, 3), fontsize=8)
                ax.set_xlabel("RMSE")
                ax.set_ylabel("R²")
                ax.set_title("Test Per-concept R² vs RMSE")
                ax.grid(True, alpha=0.3)

                # Right: R² as function of cosine similarity
                ax = axes[1]
                sc2 = ax.scatter(cos_vals, r2_vals)
                for name, x, y in zip(self.concept_list, cos_vals, r2_vals):
                    ax.annotate(name, (x, y), textcoords="offset points", xytext=(3, 3), fontsize=8)
                ax.set_xlabel("Cosine similarity")
                ax.set_ylabel("R²")
                ax.set_title("Test Per-concept R² vs cosine similarity")
                ax.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.show()


            # calculater inter-concept scores
            for i, source_concept in enumerate(self.concept_list):
                for j, target_concept in enumerate(self.concept_list):
                    if i == j:
                        continue

                    source_repr = all_concept_repr[:, i]  # représentation du concept source
                    target_repr = all_concept_repr[:, j]  # représentation du concept target
                    source_true = all_concept_labels[:, i]
                    target_true = all_concept_labels[:, j]

                    leakage_score = interconcept_leakage(source_repr, target_repr, source_true, target_true, target_continuous=True if self.concept_representation=='importance' else False)
                    pair_key = f"{source_concept} -> {target_concept}"
                    inter_concept_leakage[pair_key] = leakage_score


        if(plot_concepts):

            # # density plot of activation corresponding to all_concept_logits_artistic
            # plt.hist([value for classnumber in range(len(self.class_dict)) for value in all_concept_logits_concept1[classnumber]], bins=100, density=True, alpha=0.5, label=f'all classes')
            # plt.xlabel("Concept Activation")
            # plt.ylabel("Density")
            # plt.title(concept1+f'(f1={per_concept_f1[concept1]:.2f};leakage={per_concept_leakage[concept1]:.2f})')
            # plt.legend()
            # plt.show()

            # density plot of activation corresponding to all_concept_logits_artistic
            for classnumber in range(len(self.class_dict)):
                plt.hist(all_concept_logits_concept1[classnumber], bins=100, density=True, alpha=0.5, label=f'class pred={self.class_dict[classnumber]}')
            plt.xlabel("Concept Activation")
            plt.ylabel("Density")
            plt.title(concept1+f'(f1={per_concept_f1[concept1]:.2f};leakage={per_concept_leakage[concept1]:.2f})')
            plt.legend()
            plt.show()

            # # density plot if concept is correct or not
            # for boolean in [True, False]:
            #     plt.hist(all_concept_logits_concept1[boolean], bins=100, density=True, alpha=0.5, label=f'correct concept pred={boolean}')
            # plt.xlabel("Concept Activation")
            # plt.ylabel("Density")
            # plt.title(concept1+f'(f1={per_concept_f1[concept1]:.2f};leakage={per_concept_leakage[concept1]:.2f})')
            # plt.legend()
            # plt.show()

            for classnumber in range(len(self.class_dict)):
                plt.hist(all_concept_logits_concept2[classnumber], bins=100, density=True, alpha=0.5, label=f'class={self.class_dict[classnumber]}')
            plt.xlabel("Concept Activation")
            plt.ylabel("Density")
            plt.title(concept2+f'(f1={per_concept_f1[concept2]:.2f};leakage={per_concept_leakage[concept2]:.2f})')
            plt.legend()
            plt.show()

            # for boolean in [True, False]:
            #     plt.hist(all_concept_logits_concept2[boolean], bins=100, density=True, alpha=0.5, label=f'correct concept pred={boolean}')
            # plt.xlabel("Concept Activation")
            # plt.ylabel("Density")
            # plt.title(concept2+f'(f1={per_concept_f1[concept2]:.2f};leakage={per_concept_leakage[concept2]:.2f})')
            # plt.legend()
            # plt.show()

        #     for classnumber in range(len(self.class_dict)):
        #         plt.hist(all_concept_logits_concept3[classnumber], bins=100, density=True, alpha=0.5, label=f'class={self.class_dict[classnumber]}')
        #     plt.xlabel("Concept Activation")
        #     plt.ylabel("Density")
        #     plt.title(concept3+f'(f1={per_concept_f1[concept3]:.2f};leakage={per_concept_leakage[concept3]:.2f})')
        #     plt.legend()
        #     plt.show()

        #     for classnumber in range(len(self.class_dict)):
        #         plt.hist(all_concept_logits_concept4[classnumber], bins=100, density=True, alpha=0.5, label=f'class={self.class_dict[classnumber]}')
        #     plt.xlabel("Concept Activation")
        #     plt.ylabel("Density")
        #     plt.title(concept4+f'(f1={per_concept_f1[concept4]:.2f};leakage={per_concept_leakage[concept4]:.2f})')
        #     plt.legend()
        #     plt.show()

        return all_concept_preds, avg_loss, accuracy, f1, class_f1, concept_acc, concept_f1, per_concept_acc, per_concept_f1, concept_val_leakage, per_concept_leakage, inter_concept_leakage


    def check_nan_in_weights(self):
        for name, param in self.named_parameters():
            if param.isnan().any():
                print(f"⚠️ NaN détecté dans les poids: {name}")
                return True
        return False
    
    def save_model(self, savepath):
        if self.check_nan_in_weights() :
            print("Impossible de sauvegarder, des NaN sont présents dans le modèle.")
            return
        checkpoint = {
            'model_state_dict': self.cpu().state_dict(),  # Sauvegarde sur CPU
        }
        torch.save(checkpoint, savepath)
        print(f"Modèle sauvegardé à {savepath}")
        return 0
    
    def load_model(self, savepath):
        state_dict = torch.load(savepath, map_location=self.device)
        self.load_state_dict(state_dict['model_state_dict'])
        self.eval()
        self.to(self.device)
        return self

##################################################################################################
# inspired from
# https://github.com/enricoparisini/xai-concept-leakage/blob/main/xai_concept_leakage/metrics/mutual_information.py
# arXiv:2504.14094v2
# for the specific normalized scores defined and used in this paper, it makes no difference whether you use \(\log_2\) or the natural \(\log\) (log or ln), 
# as long as you are consistent and use the same base for all Entropy and Mutual Information calculations within the normalization.

def _entropy(x):
    """Compute Shannon entropy H(X) = -∑ p(x) log p(x)"""
    values, counts = np.unique(x, return_counts=True)
    probs = counts / counts.sum()
    return entropy(probs, base=None) # base = None -> e

def _quantile_binning(x, n_bins):
    # Compute quantile bin edges (including min and max)
    quantiles = np.linspace(0, 1, n_bins + 1)
    bins = np.unique(np.quantile(x, quantiles))
    # np.digitize expects bins to be monotonically increasing and does not include rightmost edge
    return np.digitize(x, bins[1:-1]), bins  # exclude min and max for np.digitize

def concept_task_leakage(ĉ_i, c_i, y, bins=25, target_continuous=False):
    """
    Calculate Concepts-Task Leakage (CTL) score for a single concept
    
    Args:
        ĉ_i: Array-like, predicted concept activations
        c_i: Array-like, ground-truth concept values
        y: Array-like, task discrete labels
        bins: Optional bin specification for continuous variables
    
    Returns:
        CTL score (float)
    """
    # Discretize continuous variables if needed
    if bins is not None:
        ĉ_i, _ = _quantile_binning(ĉ_i, bins)
        if(target_continuous):
            c_i, _ = _quantile_binning(c_i, bins)
        #y, _ = _quantile_binning(y, bins) # no binning necessary for discrete variables
    
    # Calculate entropy of y
    H_y = _entropy(y)
    
    if H_y == 0:
        return 0.0
    
    # Calculate normalized mutual information terms
    I_ĉy = mutual_info_score(ĉ_i, y) # base e
    I_cy = mutual_info_score(c_i, y)
    
    # Compute CTL score
    ctl = max(0, (I_ĉy - I_cy) / H_y)
    
    return ctl


def differentiable_mutual_info(x, y, sigma=None, eps=1e-12):
    """
    Differentiable mutual information (MI) estimator based on kernel density estimation (KDE).

    GOAL:
        Estimate I(X;Y) = E[ log( p(x|y) / p(x) ) ]
        in a smooth and differentiable manner without discrete bins or hard assignments.

    INPUTS
    -------
    x : torch.Tensor, shape (N,)
        Continuous-valued samples of variable X.
    y : torch.Tensor, shape (N,) or (N, C)
        Discrete class labels (ints) or one-hot encoded form for categorical variable Y.
    sigma : float or None
        Gaussian kernel bandwidth.
        If None → automatically estimated from data using Silverman's rule-of-thumb.
    eps : float
        Numerical constant to prevent numeric underflow or log(0).

    OUTPUT
    -------
    mi : torch.Tensor, scalar
        Differentiable scalar estimate of mutual information I(X;Y).
    """

    # -------------------------------------------------------------------------
    # 1. NORMALIZATION
    # -------------------------------------------------------------------------
    N = x.size(0)                                    # scalar: number of samples
    x = (x - x.mean()) / (x.std() + eps)             # (N,): zero-mean, unit-variance normalization
                                                     # → improves numerical stability for KDE

    # -------------------------------------------------------------------------
    # 2. BANDWIDTH SELECTION (SIGMA)
    # -------------------------------------------------------------------------
    # Multiple bandwidth selection strategies:
    #
    # SILVERMAN'S RULE (default): σ = 0.9 * std * N^{-1/5}
    #   - Optimal for Gaussian data
    #   - Fast and simple
    #
    # SCOTT'S RULE: σ = 1.06 * std * N^{-1/5}
    #   - More conservative (larger bandwidth)
    #   - Better for multimodal or heavy-tailed distributions
    #
    # MAD-BASED: σ = 1.06 * MAD * N^{-1/5}
    #   - Robust to outliers
    #   - MAD = median(|x - median(x)|)
    #
    if sigma is None:
        # Use Scott's rule as default (more robust than Silverman's)
        sigma = 1.06 * x.std() * (N ** (-1/5))
        
        # Alternative: MAD-based (uncomment for outlier-heavy data)
        # mad = torch.median(torch.abs(x - torch.median(x)))
        # sigma = 1.06 * mad * (N ** (-1/5))


    # make sure sigma is a tensor first (if it may be a float coming from the caller)
    if not torch.is_tensor(sigma):
        sigma = torch.tensor(sigma, device=x.device, dtype=x.dtype)
    
    sigma2 = sigma ** 2                              # scalar: σ², used in Gaussian exponent

    # -------------------------------------------------------------------------
    # 3. ONE-HOT ENCODING OF y
    # -------------------------------------------------------------------------
    if y.ndim == 1:
        num_classes = int(y.max().item() + 1)        # scalar: number of unique discrete classes
        y_onehot = F.one_hot(y, num_classes).float() # (N, C): binary class indicator matrix
    else:
        y_onehot = y.float()                         # (N, C): already one-hot
        num_classes = y_onehot.size(1)

    # -------------------------------------------------------------------------
    # 4. GAUSSIAN KERNEL MATRIX K_x
    # -------------------------------------------------------------------------
    # Pairwise squared distances: ||x_i - x_j||²
    # Shapes:
    #   x_i: (1, N)
    #   x_j: (N, 1)
    #   sq_dists: (N, N)
    x_i = x.unsqueeze(0)
    x_j = x.unsqueeze(1)
    sq_dists = (x_i - x_j) ** 2                      # (N, N): pairwise squared distances

    # Gaussian normalization constant for 1D KDE: (1 / sqrt(2πσ²))
    norm_const = 1.0 / torch.sqrt(2 * torch.pi * sigma2)

    # Compute Gaussian similarity matrix:
    #   K_x[i,j] = (1/sqrt(2πσ²)) * exp(-0.5 * (x_i - x_j)² / σ²)
    #   Represents "soft similarity" / density contribution of sample j to point i.
    K_x = norm_const * torch.exp(-0.5 * sq_dists / sigma2)   # (N, N): symmetric kernel matrix

    # -------------------------------------------------------------------------
    # 4.5. EXCLUDE SELF-SIMILARITY
    # -------------------------------------------------------------------------
    # Standard KDE practice: exclude each point's contribution to its own density estimate
    # to avoid bias. Create a mask that zeros out the diagonal.
    mask_diag = 1.0 - torch.eye(N, device=x.device, dtype=x.dtype)
    K_x = K_x * mask_diag                            # (N, N): kernel matrix with diagonal zeroed

    # -------------------------------------------------------------------------
    # 5. MARGINAL DENSITY ESTIMATION p̂(x)
    # -------------------------------------------------------------------------
    # KDE formula (excluding self):
    #   p̂(x_i) = (1/(N-1)) ∑_{j≠i} K(x_i, x_j)
    # Shapes:
    #   K_x: (N, N) with diagonal = 0
    #   p_x: (N,)
    p_x = K_x.sum(1) / (N - 1 + eps) + eps           # (N,): marginal density estimate per sample
                                                      # represents p̂(x_i)


    # plot histogram of x vs p_x
    # x_sorted, indices = torch.sort(x)
    # p_x_sorted = p_x[indices]
    # print(x_sorted.detach().cpu().numpy())
    # plt.figure(figsize=(6, 4))
    # plt.hist(x.detach().cpu().numpy(), bins=50, density=True, alpha=0.5, label='Histogram')
    # plt.plot(x_sorted.detach().cpu().numpy(), p_x_sorted.detach().cpu().numpy(), lw=2, label='Estimated density $\hat{p}(x)$')
    # plt.xlabel('x')
    # plt.ylabel('p̂(x)')
    # plt.title('Estimated marginal density vs input')
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.show()

    # -------------------------------------------------------------------------
    # 6. CONDITIONAL DENSITY ESTIMATION p̂(x|Y=c)
    # -------------------------------------------------------------------------
    # For each class c:
    #     p̂(x_i | Y=c) = (1/(N_c - δ_ic)) ∑_{j≠i, Y_j=c} K(x_i, x_j)
    # where δ_ic = 1 if sample i belongs to class c, else 0
    #
    # Implementation keeps full differentiability by summing over all pairs
    # and then weighting with the binary indicators of class membership.

    # Initialize conditional density holder
    p_x_given_y = torch.zeros_like(p_x)               # (N,): to store p̂(x_i | Y=y_i) for all i

    for c in range(num_classes):
        mask = y_onehot[:, c]                        # (N,): 1 if Y_j=c, else 0
        Nc = mask.sum()                              # scalar: number of samples in class c
        if Nc > 0:
            # Conditional numerator: ∑_{j:Y_j=c, j≠i} K(x_i, x_j)
            # Note: K_x already has diagonal zeroed, so self-contribution is excluded
            weighted_sum = (K_x * mask.unsqueeze(0)).sum(1)  # (N,): KDE sum using only class c members

            # Normalize: for samples in class c, divide by (N_c - 1) to exclude self
            # For samples not in class c, divide by N_c (no self to exclude)
            # We use (Nc - mask) to handle both cases:
            #   - If sample i is in class c: Nc - 1
            #   - If sample i is not in class c: Nc
            norm_factor = Nc - mask + eps
            cond_density = weighted_sum / norm_factor         # (N,): p̂(x_i | Y=c)

            # Assign to samples that actually belong to class c
            p_x_given_y += cond_density * mask                 # (N,): keep only p̂(x_i | y_i)

    # -------------------------------------------------------------------------
    # 7. MUTUAL INFORMATION COMPUTATION
    # -------------------------------------------------------------------------
    # Mutual Information Definition:
    #   I(X;Y) = E[ log p(x|y) - log p(x) ]
    #
    # In empirical (sample-based) form:
    #   Î = (1/N) ∑_i log( p̂(x_i|y_i) / p̂(x_i) )
    #
    # Since all kernel-based estimates are differentiable,
    # this operation maintains differentiability for gradient-based learning.

    log_ratio = torch.log(p_x_given_y + eps) - torch.log(p_x + eps)  # (N,): elementwise log ratio
    mi = log_ratio.mean()                                            # scalar: average MI estimate

    # -------------------------------------------------------------------------
    # 8. OUTPUT SUMMARY
    # -------------------------------------------------------------------------
    #
    # OUTPUT: scalar tensor ≈ I(X;Y)
    #
    # INTERPRETATION:
    #   - mi → 0 when X and Y are independent (p̂(x|y) ≈ p̂(x))
    #   - mi increases as class-conditional distributions diverge
    #
    # DIFFERENTIABILITY:
    #   - gradient flows through Gaussian kernels (useful for deep MI minimization/maximization)
    #
    # IMPROVEMENTS vs. original:
    #   - Self-similarity excluded from KDE (standard practice, reduces bias)
    #   - Proper normalization: (N-1) for marginal, (N_c - δ_ic) for conditional
    #
    # ASSUMPTIONS:
    #   * Samples (x_i, y_i) are i.i.d. draws from some joint p(x, y)
    #   * Gaussian KDE adequately captures local structure
    #   * 1D X (for multidimensional X, replace distance with squared L2 norm)
    #

    return mi


def pytorch_concept_task_leakage_differentiable(ĉ_i, c_i, y, sigma=None, target_continuous=False, bins=25):
    """
    Differentiable CTL approximation.
    """
    # Use continuous MI approximation (differentiable)
    I_ĉy = differentiable_mutual_info(ĉ_i, y, sigma)

    if(target_continuous):
        I_cy = differentiable_mutual_info(c_i, y, sigma)
    else:
        # True concepts are discrete - no grad needed

        if torch.is_tensor(c_i):
            c_i_np = c_i.detach().cpu().numpy()
        else:
            c_i_np = c_i
            
        if torch.is_tensor(y):
            y_np = y.detach().cpu().numpy()
        else:
            y_np = y

        # sklearn mutual_info_score returns float
        I_cy = mutual_info_score(c_i_np, y_np)
        
        # Convert back to tensor (same device as I_ĉy)
        I_cy = torch.tensor(I_cy, device=ĉ_i.device, dtype=ĉ_i.dtype)
    
    # Calculate entropy of y (constant so no grad)
    y_np = y.detach().cpu().numpy() if torch.is_tensor(y) else np.asarray(y)
    H_y = _entropy(y_np)  # Returns numpy float
    
    if H_y == 0 or H_y < 1e-8:
        return torch.tensor(0., device=ĉ_i.device, dtype=ĉ_i.dtype)
    
    # Convert H_y to tensor
    H_y_tensor = torch.tensor(H_y, device=ĉ_i.device, dtype=ĉ_i.dtype)
    
    #ctl = torch.clamp((I_ĉy - I_cy)/ H_y.detach(), min=0.0)  
    leakage_term = (I_ĉy - I_cy) / H_y_tensor
    ctl = leakage_term ** 2 # Squared instead of clamped to not promote less information contained in learned concepts than true concepts

    return ctl


def interconcept_leakage(ĉ_i, ĉ_j, c_i, c_j, bins=25, target_continuous=True):
    """
    Calculate ICL_{ij} score for a concept pair.
    
    Args:
        ĉ_i: Predicted concept i (array-like)
        ĉ_j: Predicted concept j (array-like)
        c_i: Ground-truth concept i (array-like)
        c_j: Ground-truth concept j (array-like)
        bins: Optional, number of bins or bin edges for discretization (for continuous variables)
    Returns:
        ICL_{ij} score (float)
    """
    # Discretize if needed
    if bins is not None:
        ĉ_i, _ = _quantile_binning(ĉ_i, bins)
        ĉ_j, _ = _quantile_binning(ĉ_j, bins)
        if(target_continuous):
            c_i, _ = _quantile_binning(c_i, bins)
            c_j, _ = _quantile_binning(c_j, bins)
    
    # Compute entropies
    H_ĉi = _entropy(ĉ_i)
    H_ĉj = _entropy(ĉ_j)
    H_ci = _entropy(c_i)
    H_cj = _entropy(c_j)
    
    # Avoid division by zero
    denom_learned = np.sqrt(H_ĉi * H_ĉj) if (H_ĉi > 0) and (H_ĉj > 0) else 1e-10
    denom_true = np.sqrt(H_ci * H_cj) if (H_ci > 0) and (H_cj > 0) else 1e-10
    
    # Compute mutual informations
    I_ĉ = mutual_info_score(ĉ_i, ĉ_j)
    I_c = mutual_info_score(c_i, c_j)
    
    # Normalized MI
    nmi_learned = I_ĉ / denom_learned
    nmi_true = I_c / denom_true
    
    # ICL score
    icl = max(0, nmi_learned - nmi_true)
    return icl

def get_base_model(base = None) :
    if base is None :
        base = input("Enter base model name : ")
    if base == "clip":
        base_model_name = "openai/clip-vit-base-patch32"
        base_model = CLIPModel.from_pretrained(base_model_name)
        processor = CLIPProcessor.from_pretrained(base_model_name)
    elif base == "blip":
        base_model_name = "Salesforce/blip-image-captioning-base"
        base_model = BlipModel.from_pretrained(base_model_name)
        processor = BlipProcessor.from_pretrained(base_model_name)
    return base_model, processor

##################################################################################################

def save_fig_to_dbfs(fig, filename, base_path):
    os.makedirs(base_path, exist_ok=True)
    full_path = os.path.join(base_path, filename)
    fig.savefig(full_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved to DBFS: {full_path}")

def plot_concept_to_class_weights(model, num_top_concepts=6, figsize=(12, 10), combine_type='logits'):
    """
    Plot the weights of the final classifier layer connecting concepts to classes.
    
    Args:
        model: CBM_binary model with classifier weights
        num_top_concepts: Number of top concepts to display per class
        figsize: Figure dimensions
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap

    weights = model.classifier.weight.detach().cpu().numpy()
    class_names = [model.class_dict[i] for i in range(model.num_classes)]
    concept_names = model.concept_list

    if(combine_type=='concat'):
        concept_names = [concept.replace('concept_text_','') for concept in concept_names[:int(len(concept_names)/2)]]

    # Collect top concept indices for each class
    top_concept_indices = [np.argsort(weights[c])[-num_top_concepts:][::-1] 
                          for c in range(model.num_classes)]

    # Create unique concept list maintaining order
    unique_concepts = []
    for indices in top_concept_indices:
        for idx in indices:
            if idx not in unique_concepts:
                unique_concepts.append(idx)

    # Build heatmap matrix
    heatmap_matrix = np.array([[weights[c, idx] for idx in unique_concepts] 
                              for c in range(model.num_classes)])

    # Formatting
    colors = ["darkblue", "blue", "lightblue", "white", "lightcoral", "red", "darkred"]
    cmap = LinearSegmentedColormap.from_list("custom_diverging", colors, N=256)
    formatted_names = [name.replace('concept_text_', '').replace('concept_image_', '').replace('concept_','') for name in np.array(concept_names)[unique_concepts]]

    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        heatmap_matrix,
        cmap=cmap,
        center=0,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Weight Value"},
        linewidths=0.5,
        xticklabels=formatted_names,
        yticklabels=class_names
    )
    plt.xticks(rotation=45, ha="right")
    plt.title("Top Concepts for Each Class by Weight Magnitude", fontsize=16)
    plt.tight_layout()
    plt.show()

def compute_leakage_matrix(inter_concept_leakage, concept_list):
    
    """Define leakage matrix from interconcept leakage scores """

    n_concepts = len(concept_list)
    concept_list_stripped = [c.replace("concept_", "") for c in concept_list]

    leakage_matrix = pd.DataFrame(
        np.zeros((len(concept_list_stripped), len(concept_list_stripped))),
        index=concept_list_stripped,
        columns=concept_list_stripped
    )

    inter_concept_leakage_stripped = {
        pair_key.replace("concept_", ""): score
        for pair_key, score in inter_concept_leakage.items()
    }

    for pair_key, score in inter_concept_leakage_stripped.items():
        source, target = pair_key.split(" -> ")
        if source in concept_list_stripped and target in concept_list_stripped:
            leakage_matrix.loc[source, target] = score
    
    return leakage_matrix

def plot_interconcept_leakage_heatmaps(inter_concept_leakage, concept_list):

    n_concepts = len(concept_list)
    
    concept_list_stripped = [c.replace("concept_", "") for c in concept_list]

    filtered_concepts_N24 = {
        class_name: [c for c in concepts if c in concept_list_stripped]
        for class_name, concepts in concepts_N24.items()
    }

    concept_to_class = {
        concept: class_name
        for class_name, concepts in filtered_concepts_N24.items()
        for concept in concepts
    }
    
    class_list = sorted(filtered_concepts_N24.keys())

    leakage_matrix = compute_leakage_matrix(inter_concept_leakage, concept_list)

    ### Heatmap intra-classes
    for class_name, concepts in filtered_concepts_N24.items():
        sub_matrix = leakage_matrix.loc[concepts, concepts]

        plt.figure(figsize=(max(6, len(concepts) * 0.5), max(5, len(concepts) * 0.5)))
        sns.set(font_scale=0.8)
        ax = sns.heatmap(
            sub_matrix,
            annot=False,
            cmap="Reds",
            cbar_kws={'label': 'Leakage Score'},
            linewidths=0.3,
            linecolor='lightgray'
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
        plt.title(f"Intra-Class Leakage: {class_name}")
        plt.xlabel("Target Concept")
        plt.ylabel("Source Concept")
        plt.tight_layout()
        plt.show()

    ### Heatmap inter-classes (moyenne des leakage scores)
    class_matrix = pd.DataFrame(
        np.zeros((len(class_list), len(class_list))),
        index=class_list,
        columns=class_list
    )
    count_matrix = pd.DataFrame(
        np.zeros((len(class_list), len(class_list))),
        index=class_list,
        columns=class_list
    )

    for src_concept in concept_list_stripped:
        for tgt_concept in concept_list_stripped:
            if src_concept != tgt_concept:
                src_class = concept_to_class.get(src_concept)
                tgt_class = concept_to_class.get(tgt_concept)
                if src_class and tgt_class:
                    value = leakage_matrix.loc[src_concept, tgt_concept]
                    class_matrix.loc[src_class, tgt_class] += value
                    count_matrix.loc[src_class, tgt_class] += 1

    with np.errstate(divide='ignore', invalid='ignore'):
        avg_matrix = class_matrix / count_matrix
        avg_matrix = avg_matrix.fillna(0)

    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.0)
    ax = sns.heatmap(
        avg_matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        cbar_kws={'label': 'Avg Leakage Score'},
        linewidths=0.5,
        linecolor='gray'
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.title("Inter-Class Leakage (Average)")
    plt.xlabel("Target Class")
    plt.ylabel("Source Class")
    plt.tight_layout()
    plt.show()


def plot_interconcept_leakage_by_cluster(inter_concept_leakage, concept_list, cluster_json_path):

    # Charger les clusters
    with open(cluster_json_path, "r") as f:
        cluster_dict = json.load(f)

    # Nettoyer les noms
    # concept_list_stripped = [c.replace("concept_", "") for c in concept_list]
    concept_list_stripped = concept_list

    # Filtrer les clusters pour ne garder que les concepts du modèle
    filtered_clusters = {
        cluster_name: [c for c in concepts if c in concept_list_stripped]
        for cluster_name, concepts in cluster_dict.items()
    }

    # Supprimer les clusters vides
    filtered_clusters = {k: v for k, v in filtered_clusters.items() if v}

    # Création d'un mapping concept -> cluster
    concept_to_cluster = {
        concept: cluster_name
        for cluster_name, concepts in filtered_clusters.items()
        for concept in concepts
    }

    cluster_list = sorted(filtered_clusters.keys())

    # Initialiser la matrice de leakage
    leakage_matrix = pd.DataFrame(
        np.zeros((len(concept_list_stripped), len(concept_list_stripped))),
        index=concept_list_stripped,
        columns=concept_list_stripped
    )

    # Nettoyer les clés de `inter_concept_leakage`
    # inter_concept_leakage_stripped = {
    #     pair_key.replace("concept_", ""): score
    #     for pair_key, score in inter_concept_leakage.items()
    # }
    inter_concept_leakage_stripped = inter_concept_leakage

    for pair_key, score in inter_concept_leakage_stripped.items():
        source, target = pair_key.split(" -> ")
        if source in concept_list_stripped and target in concept_list_stripped:
            leakage_matrix.loc[source, target] = score

    ### Heatmap intra-cluster
    for cluster_name, concepts in filtered_clusters.items():
        sub_matrix = leakage_matrix.loc[concepts, concepts]
        
        plt.figure(figsize=(max(6, len(concepts) * 0.5), max(5, len(concepts) * 0.5)))
        sns.set(font_scale=0.8)
        ax = sns.heatmap(
            sub_matrix,
            annot=False,
            cmap="Reds",
            cbar_kws={'label': 'Leakage Score'},
            linewidths=0.3,
            linecolor='lightgray'
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
        plt.title(f"Intra-Cluster Leakage: {cluster_name}")
        plt.xlabel("Target Concept")
        plt.ylabel("Source Concept")
        plt.tight_layout()
        plt.show()

    ### Heatmap inter-cluster (moyenne des scores)
    cluster_matrix = pd.DataFrame(
        np.zeros((len(cluster_list), len(cluster_list))),
        index=cluster_list,
        columns=cluster_list
    )
    count_matrix = pd.DataFrame(
        np.zeros((len(cluster_list), len(cluster_list))),
        index=cluster_list,
        columns=cluster_list
    )

    for src_concept in concept_list_stripped:
        for tgt_concept in concept_list_stripped:
            if src_concept != tgt_concept:
                src_cluster = concept_to_cluster.get(src_concept)
                tgt_cluster = concept_to_cluster.get(tgt_concept)
                if src_cluster and tgt_cluster:
                    value = leakage_matrix.loc[src_concept, tgt_concept]
                    cluster_matrix.loc[src_cluster, tgt_cluster] += value
                    count_matrix.loc[src_cluster, tgt_cluster] += 1

    with np.errstate(divide='ignore', invalid='ignore'):
        avg_cluster_matrix = cluster_matrix / count_matrix
        avg_cluster_matrix = avg_cluster_matrix.fillna(0)

    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.0)
    ax = sns.heatmap(
        avg_cluster_matrix,
        annot=False,
        fmt=".2f",
        cmap="Blues",
        cbar_kws={'label': 'Avg Leakage Score'},
        linewidths=0.5,
        linecolor='gray'
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.title("Inter-Cluster Leakage (Average)")
    plt.xlabel("Target Cluster")
    plt.ylabel("Source Cluster")
    plt.tight_layout()
    plt.show()

def plot_leakage_graph_for_class(model, inter_concept_leakage, class_idx, num_top_concepts=6):
    import networkx as nx
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    weights = model.classifier.weight.detach().cpu().numpy()
    concept_names = np.array(model.concept_list)
    class_name = model.class_dict[class_idx]

    top_indices = np.argsort(weights[class_idx])[-num_top_concepts:][::-1]
    top_concepts = concept_names[top_indices]
    top_weights = weights[class_idx, top_indices]

    concept_stripped = [c.replace('concept_', '') for c in top_concepts]
    leakage_keys = {k.replace('concept_', ''): v for k, v in inter_concept_leakage.items()}

    G = nx.Graph()  # <- Non orienté = traits sans flèches

    for i, concept in enumerate(concept_stripped):
        G.add_node(concept, weight=top_weights[i])

    # Ajouter les arêtes symétriquement (même si on part de données dirigées)
    for src in concept_stripped:
        for tgt in concept_stripped:
            if src != tgt:
                key_forward = f"{src} -> {tgt}"
                key_backward = f"{tgt} -> {src}"
                score = leakage_keys.get(key_forward, leakage_keys.get(key_backward, 0))
                if score > 0:
                    G.add_edge(src, tgt, weight=score)

    pos = nx.spring_layout(G, seed=42)
    node_sizes = [abs(G.nodes[n]['weight']) * 1000 for n in G.nodes]
    edge_widths = [G[u][v]['weight'] * 10 for u, v in G.edges]

    plt.figure(figsize=(8, 6))

    # Traits sans flèches
    nx.draw_networkx_edges(
        G, pos,
        width=edge_widths,
        edge_color='gray',
        alpha=0.7,
        arrows=False  # n'a d'effet que si G est non orienté
    )

    # Noeuds
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color='skyblue',
        edgecolors='black',
        linewidths=1
    )

    # Étiquettes
    nx.draw_networkx_labels(G, pos, font_size=10)

    # Légende
    legend_elements = [
        Patch(color='skyblue', label='Concept (taille = importance)'),
        Line2D([0], [0], color='gray', lw=2, label='Fuite entre concepts (épaisseur = score)')
    ]
    plt.legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.title(f"Concept Leakage Graph for Class: {class_name}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_leakage_graph_by_activation(
    concept_activations, 
    interconcept_leakage, 
    concept_list, 
    per_concept_acc, 
    per_concept_f1, 
    num_top_concepts=6
):

    # Étape 1 : fréquence d’activation
    if concept_activations.ndim != 2:
        raise ValueError("concept_activations should be (n_samples, n_concepts)")
    
    frequencies = np.mean(concept_activations, axis=0)

    # Nettoyage noms
    def clean_name(c):
        return (
            str(c)
            .replace("concept_", "")
            .replace("text_", "")
            .replace("image_", "")
            .strip()
        )

    concept_names = np.array([clean_name(c) for c in concept_list])

    # Séparation TEXT / IMAGE
    is_text = np.array(["text" in str(c).lower() for c in concept_list])
    is_image = ~is_text

    leakage_keys = {}
    for k, v in interconcept_leakage.items():
        a, b = [clean_name(x) for x in k.split("->")]
        leakage_keys[f"{a.strip()} -> {b.strip()}"] = v

    leakage_sums = []
    for concept in concept_names:
        total = 0
        for other in concept_names:
            if concept != other:
                key_fwd = f"{concept} -> {other}"
                key_bwd = f"{other} -> {concept}"
                total += leakage_keys.get(key_fwd, leakage_keys.get(key_bwd, 0))
        leakage_sums.append(total)

    leakage_sums = np.array(leakage_sums)

    acc_values = []
    for orig_name in concept_list:
        key = str(orig_name)
        acc_values.append(per_concept_acc[key])

    acc_values = np.array(acc_values)

    # Indices texte/image
    text_indices = np.where(is_text)[0]
    image_indices = np.where(is_image)[0]

    # Classements séparés
    sorted_text  = text_indices[np.argsort(acc_values[text_indices])]
    sorted_image = image_indices[np.argsort(acc_values[image_indices])]

    # Worst = 20 moins bons / Best = 20 meilleurs
    worst_text = sorted_text[:20]
    best_text  = sorted_text[-20:]

    worst_image = sorted_image[:20]
    best_image  = sorted_image[-20:]

    plt.figure(figsize=(22, 12))

    # Couleurs (pas de rouge)
    text_color = "dodgerblue"
    image_color = "darkorange"

    # Points texte
    plt.scatter(
        frequencies[text_indices],
        leakage_sums[text_indices],
        c=text_color, s=120, edgecolors="black", label="Text concepts"
    )

    # Points image
    plt.scatter(
        frequencies[image_indices],
        leakage_sums[image_indices],
        c=image_color, s=120, edgecolors="black", label="Image concepts"
    )

    worst_all = set(list(worst_text) + list(worst_image))

    for i in range(len(concept_list)):
        color = "red" if i in worst_all else "black"
        plt.text(
            frequencies[i] + 0.003,
            leakage_sums[i],
            concept_names[i],
            fontsize=9,
            color=color
        )

    # Axes
    plt.xlabel("Activation frequency")
    plt.ylabel("Total leakage score")
    plt.title("Leakage score by Concept Activation — Text vs Image")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_leakage_score_by_co_activation(concept_activations, interconcept_leakage, concept_list):
    """
    interconcept_leakage  : dict {"concept_a->concept_b": leakage score, ...}
    """

    coactivation_matrix = cosine_similarity(concept_activations)

    concept_names = [str(c).replace("concept_", "").strip() for c in concept_list]
    n_concepts = len(concept_names)

    leakage_keys = {}
    for k, v in interconcept_leakage.items():
        parts = re.split(r'\s*->\s*', k)
        if len(parts) == 2:
            a, b = parts
            a = a.replace("concept_", "").strip()
            b = b.replace("concept_", "").strip()
            leakage_keys[f"{a}->{b}"] = v

    xs, ys, labels = [], [], []

    for i in range(n_concepts):
        for j in range(i+1, n_concepts):
            a, b = concept_names[i], concept_names[j]
            coact = float(coactivation_matrix[i, j])
            leakage = leakage_keys.get(f"{a}->{b}", leakage_keys.get(f"{b}->{a}", 0.0))
            xs.append(coact)
            ys.append(leakage)
            labels.append(f"{a} ↔ {b}")

    xs = np.array(xs)
    ys = np.array(ys)

    # Scatter plot (points petits et sans labels)
    plt.figure(figsize=(14, 6))
    plt.scatter(xs, ys, s=30, facecolors="lightblue", edgecolors="black")

    plt.xlabel("Concept co-activation")
    plt.ylabel("Leakage score")
    plt.title("Leakage score by co-activation")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

def heatmap_leakage_score_by_co_activation(concept_activations, interconcept_leakage, concept_list):

    coactivation_matrix = cosine_similarity(concept_activations)

    concept_names = [str(c).replace("concept_", "").strip() for c in concept_list]
    n_concepts = len(concept_names)

    leakage_matrix = np.zeros((n_concepts, n_concepts))

    for k, v in interconcept_leakage.items():
        parts = re.split(r'\s*->\s*', k)
        if len(parts) == 2:
            a, b = parts
            a = a.replace("concept_", "").strip()
            b = b.replace("concept_", "").strip()
            if a in concept_names and b in concept_names:
                i, j = concept_names.index(a), concept_names.index(b)
                leakage_matrix[i, j] = v
    
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        leakage_matrix,
        xticklabels=concept_names,
        yticklabels=concept_names,
        cmap="Reds",
        annot=False,
        fmt=".2f",
        cbar_kws={"label": "Leakage score"}
    )

    ax.set_title("Inter-concept leakage heatmap", fontsize=14, pad=12)
    ax.set_xlabel("To concept", fontsize=12)
    ax.set_ylabel("From concept", fontsize=12)

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def compute_leakage_matrix(inter_concept_leakage, concept_list):
    
    """Define leakage matrix from interconcept leakage scores """

    n_concepts = len(concept_list)
    concept_list_stripped = [c.replace("concept_", "") for c in concept_list]

    leakage_matrix = pd.DataFrame(
        np.zeros((len(concept_list_stripped), len(concept_list_stripped))),
        index=concept_list_stripped,
        columns=concept_list_stripped
    )

    inter_concept_leakage_stripped = {
        pair_key.replace("concept_", ""): score
        for pair_key, score in inter_concept_leakage.items()
    }

    for pair_key, score in inter_concept_leakage_stripped.items():
        source, target = pair_key.split(" -> ")
        if source in concept_list_stripped and target in concept_list_stripped:
            leakage_matrix.loc[source, target] = score
    
    return leakage_matrix

def plot_interconcept_leakage_heatmaps(inter_concept_leakage, concept_list):

    def color_for_concept(name):
        if "text_" in name:
            return "blue"
        elif "image_" in name:
            return "orange"
        else:
            return "black"
        
    n_concepts = len(concept_list)
    
    concept_list_stripped = [c.replace("concept_", "") for c in concept_list]

    filtered_concepts_N24 = {
        class_name: [c for c in concepts if c in concept_list_stripped]
        for class_name, concepts in concepts_N24.items()
    }

    concept_to_class = {
        concept: class_name
        for class_name, concepts in filtered_concepts_N24.items()
        for concept in concepts
    }
    
    class_list = sorted(filtered_concepts_N24.keys())

    leakage_matrix = compute_leakage_matrix(inter_concept_leakage, concept_list)

    ### Heatmap intra-classes
    for class_name, concepts in filtered_concepts_N24.items():
        if len(concepts) == 0:
            print(f"[INFO] Classe ignorée (aucun concept) : {class_name}")
            continue

        sub_matrix = leakage_matrix.loc[concepts, concepts]

        if sub_matrix.size == 0:
            print(f"[INFO] Matrice vide ignorée pour la classe : {class_name}")
            continue

        plt.figure(figsize=(max(6, len(concepts) * 0.5), max(5, len(concepts) * 0.5)))
        sns.set(font_scale=0.8)
        ax = sns.heatmap(
            sub_matrix,
            annot=False,
            cmap="Reds",
            cbar_kws={'label': 'Leakage Score'},
            linewidths=0.3,
            linecolor='lightgray'
        )
        
        # Colorizing labels
        xticks = ax.get_xticklabels()
        yticks = ax.get_yticklabels()

        for label in xticks:
            label.set_color(color_for_concept(label.get_text()))
        for label in yticks:
            label.set_color(color_for_concept(label.get_text()))

        ax.set_xticklabels(xticks, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(yticks, rotation=0, fontsize=9)

        plt.title(f"Intra-Class Leakage: {class_name}")
        plt.xlabel("Target Concept")
        plt.ylabel("Source Concept")
        plt.tight_layout()
        plt.show()

    ### Heatmap inter-classes (moyenne des leakage scores)
    class_matrix = pd.DataFrame(
        np.zeros((len(class_list), len(class_list))),
        index=class_list,
        columns=class_list
    )
    count_matrix = pd.DataFrame(
        np.zeros((len(class_list), len(class_list))),
        index=class_list,
        columns=class_list
    )

    for src_concept in concept_list_stripped:
        for tgt_concept in concept_list_stripped:
            if src_concept != tgt_concept:
                src_class = concept_to_class.get(src_concept)
                tgt_class = concept_to_class.get(tgt_concept)
                if src_class and tgt_class:
                    value = leakage_matrix.loc[src_concept, tgt_concept]
                    class_matrix.loc[src_class, tgt_class] += value
                    count_matrix.loc[src_class, tgt_class] += 1

    with np.errstate(divide='ignore', invalid='ignore'):
        avg_matrix = class_matrix / count_matrix
        avg_matrix = avg_matrix.fillna(0)

    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.0)
    ax = sns.heatmap(
        avg_matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        cbar_kws={'label': 'Avg Leakage Score'},
        linewidths=0.5,
        linecolor='gray'
    )

    # Colorize class labels (optional, same rule)
    xticks = ax.get_xticklabels()
    yticks = ax.get_yticklabels()

    for label in xticks:
        label.set_color(color_for_concept(label.get_text()))
    for label in yticks:
        label.set_color(color_for_concept(label.get_text()))

    ax.set_xticklabels(xticks, rotation=45, ha='right')
    ax.set_yticklabels(yticks, rotation=0)

    plt.title("Inter-Class Leakage (Average)")
    plt.xlabel("Target Class")
    plt.ylabel("Source Class")
    plt.tight_layout()
    plt.show()


def plot_interconcept_leakage_by_cluster(inter_concept_leakage, concept_list, cluster_json_path):

    # Charger les clusters
    with open(cluster_json_path, "r") as f:
        cluster_dict = json.load(f)

    # Nettoyer les noms
    # concept_list_stripped = [c.replace("concept_", "") for c in concept_list]
    concept_list_stripped = concept_list

    # Filtrer les clusters pour ne garder que les concepts du modèle
    filtered_clusters = {
        cluster_name: [c for c in concepts if c in concept_list_stripped]
        for cluster_name, concepts in cluster_dict.items()
    }

    # Supprimer les clusters vides
    filtered_clusters = {k: v for k, v in filtered_clusters.items() if v}

    # Création d'un mapping concept -> cluster
    concept_to_cluster = {
        concept: cluster_name
        for cluster_name, concepts in filtered_clusters.items()
        for concept in concepts
    }

    cluster_list = sorted(filtered_clusters.keys())

    # Initialiser la matrice de leakage
    leakage_matrix = pd.DataFrame(
        np.zeros((len(concept_list_stripped), len(concept_list_stripped))),
        index=concept_list_stripped,
        columns=concept_list_stripped
    )

    # Nettoyer les clés de `inter_concept_leakage`
    # inter_concept_leakage_stripped = {
    #     pair_key.replace("concept_", ""): score
    #     for pair_key, score in inter_concept_leakage.items()
    # }
    inter_concept_leakage_stripped = inter_concept_leakage

    for pair_key, score in inter_concept_leakage_stripped.items():
        source, target = pair_key.split(" -> ")
        if source in concept_list_stripped and target in concept_list_stripped:
            leakage_matrix.loc[source, target] = score

    ### Heatmap intra-cluster
    for cluster_name, concepts in filtered_clusters.items():
        sub_matrix = leakage_matrix.loc[concepts, concepts]
        
        plt.figure(figsize=(max(6, len(concepts) * 0.5), max(5, len(concepts) * 0.5)))
        sns.set(font_scale=0.8)
        ax = sns.heatmap(
            sub_matrix,
            annot=False,
            cmap="Reds",
            cbar_kws={'label': 'Leakage Score'},
            linewidths=0.3,
            linecolor='lightgray'
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
        plt.title(f"Intra-Cluster Leakage: {cluster_name}")
        plt.xlabel("Target Concept")
        plt.ylabel("Source Concept")
        plt.tight_layout()
        plt.show()

    ### Heatmap inter-cluster (moyenne des scores)
    cluster_matrix = pd.DataFrame(
        np.zeros((len(cluster_list), len(cluster_list))),
        index=cluster_list,
        columns=cluster_list
    )
    count_matrix = pd.DataFrame(
        np.zeros((len(cluster_list), len(cluster_list))),
        index=cluster_list,
        columns=cluster_list
    )

    for src_concept in concept_list_stripped:
        for tgt_concept in concept_list_stripped:
            if src_concept != tgt_concept:
                src_cluster = concept_to_cluster.get(src_concept)
                tgt_cluster = concept_to_cluster.get(tgt_concept)
                if src_cluster and tgt_cluster:
                    value = leakage_matrix.loc[src_concept, tgt_concept]
                    cluster_matrix.loc[src_cluster, tgt_cluster] += value
                    count_matrix.loc[src_cluster, tgt_cluster] += 1

    with np.errstate(divide='ignore', invalid='ignore'):
        avg_cluster_matrix = cluster_matrix / count_matrix
        avg_cluster_matrix = avg_cluster_matrix.fillna(0)

    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.0)
    ax = sns.heatmap(
        avg_cluster_matrix,
        annot=False,
        fmt=".2f",
        cmap="Blues",
        cbar_kws={'label': 'Avg Leakage Score'},
        linewidths=0.5,
        linecolor='gray'
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.title("Inter-Cluster Leakage (Average)")
    plt.xlabel("Target Cluster")
    plt.ylabel("Source Cluster")
    plt.tight_layout()
    plt.show()

def plot_leakage_graph_for_class(model, inter_concept_leakage, class_idx, num_top_concepts=6):
    import networkx as nx
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    weights = model.classifier.weight.detach().cpu().numpy()
    concept_names = np.array(model.concept_list)
    class_name = model.class_dict[class_idx]

    top_indices = np.argsort(weights[class_idx])[-num_top_concepts:][::-1]
    top_concepts = concept_names[top_indices]
    top_weights = weights[class_idx, top_indices]

    concept_stripped = [c.replace('concept_', '') for c in top_concepts]
    leakage_keys = {k.replace('concept_', ''): v for k, v in inter_concept_leakage.items()}

    G = nx.Graph()  # <- Non orienté = traits sans flèches

    for i, concept in enumerate(concept_stripped):
        G.add_node(concept, weight=top_weights[i])

    # Ajouter les arêtes symétriquement (même si on part de données dirigées)
    for src in concept_stripped:
        for tgt in concept_stripped:
            if src != tgt:
                key_forward = f"{src} -> {tgt}"
                key_backward = f"{tgt} -> {src}"
                score = leakage_keys.get(key_forward, leakage_keys.get(key_backward, 0))
                if score > 0:
                    G.add_edge(src, tgt, weight=score)

    pos = nx.spring_layout(G, seed=42)
    node_sizes = [abs(G.nodes[n]['weight']) * 1000 for n in G.nodes]
    edge_widths = [G[u][v]['weight'] * 10 for u, v in G.edges]

    plt.figure(figsize=(8, 6))

    # Traits sans flèches
    nx.draw_networkx_edges(
        G, pos,
        width=edge_widths,
        edge_color='gray',
        alpha=0.7,
        arrows=False  # n'a d'effet que si G est non orienté
    )

    # Noeuds
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color='skyblue',
        edgecolors='black',
        linewidths=1
    )

    # Étiquettes
    nx.draw_networkx_labels(G, pos, font_size=10)

    # Légende
    legend_elements = [
        Patch(color='skyblue', label='Concept (taille = importance)'),
        Line2D([0], [0], color='gray', lw=2, label='Fuite entre concepts (épaisseur = score)')
    ]
    plt.legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.title(f"Concept Leakage Graph for Class: {class_name}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_tast_leakage_by_activation_similarity(concept_activations, concept_list, per_concept_acc, task_concept_leakage):
    if concept_activations.ndim != 2:
        raise ValueError("concept_activations should be 2D (n_samples, n_concepts)")

    n_samples, n_concepts = concept_activations.shape
    print(f"n_samples: {n_samples}")
    print(f"n_concepts: {n_concepts}")
    concept_names = np.array([c.replace("concept_", "") for c in concept_list])

    cos_sim_matrix = cosine_similarity(concept_activations.T)
    print(f"cos_sim_matrix: {np.shape(cos_sim_matrix)}")

    abs_cos_sim = np.abs(cos_sim_matrix)
    np.fill_diagonal(abs_cos_sim, np.nan)
    mean_abs_cos_sim = np.nanmean(abs_cos_sim, axis=1)

    task_leak_values = np.array(
        [task_concept_leakage[f"concept_{c}"] for c in concept_names]
    )

    plt.figure(figsize=(14, 8))

    plt.scatter(
        mean_abs_cos_sim,
        task_leak_values,
        s=80,
        c="steelblue",
        edgecolors="black"
    )

    for i in range(n_concepts):
        plt.text(
            mean_abs_cos_sim[i] + 0.002,
            task_leak_values[i],
            concept_names[i],
            fontsize=9
        )

    plt.xlabel("Mean absolute cosine similarity with other concepts")
    plt.ylabel("Task leakage score")
    plt.title("Task Leakage vs Concept Activation Similarity")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_task_leakage_by_weight_similarity(model,concept_list,per_concept_acc,task_concept_leakage):
    concept_names = np.array([c.replace("concept_", "") for c in concept_list])

    # === Récupération et normalisation des poids ===
    if hasattr(model, "concept_layer"):
        concept_weight_matrix = (
            model.concept_layer.weight.detach().cpu().numpy()
        )
        norm_weights = concept_weight_matrix / np.linalg.norm(
            concept_weight_matrix, axis=1, keepdims=True
        )

    else:
        concept_weight_matrix_text = (
            model.concept_layer_text.weight.detach().cpu().numpy()
        )
        norm_weights_text = concept_weight_matrix_text / np.linalg.norm(
            concept_weight_matrix_text, axis=1, keepdims=True
        )

        concept_weight_matrix_image = (
            model.concept_layer_image.weight.detach().cpu().numpy()
        )
        norm_weights_image = concept_weight_matrix_image / np.linalg.norm(
            concept_weight_matrix_image, axis=1, keepdims=True
        )

        # Si les concepts sont partagés texte/image, concaténation
        norm_weights = np.concatenate(
            [norm_weights_text, norm_weights_image],
            axis=1
        )

    n_concepts = norm_weights.shape[0]
    print(f"n_concepts: {n_concepts}")

    # === Cosine similarity entre poids ===
    cos_sim_matrix = cosine_similarity(norm_weights)
    print(f"cos_sim_matrix shape: {cos_sim_matrix.shape}")

    abs_cos_sim = np.abs(cos_sim_matrix)
    np.fill_diagonal(abs_cos_sim, np.nan)
    mean_abs_cos_sim = np.nanmean(abs_cos_sim, axis=1)

    # === Task leakage ===
    task_leak_values = np.array(
        [task_concept_leakage[f"concept_{c}"] for c in concept_names]
    )

    # === Plot ===
    plt.figure(figsize=(14, 8))
    plt.scatter(
        mean_abs_cos_sim,
        task_leak_values,
        s=80,
        c="steelblue",
        edgecolors="black"
    )

    for i in range(n_concepts):
        plt.text(
            mean_abs_cos_sim[i] + 0.002,
            task_leak_values[i],
            concept_names[i],
            fontsize=9
        )

    plt.xlabel("Mean absolute cosine similarity between concept weights")
    plt.ylabel("Task leakage score")
    plt.title("Task Leakage vs Concept Weight Similarity")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_leakage_bar_graph_by_activation(concept_activations, interconcept_leakage, concept_list, per_concept_acc, task_concept_leakage, save_to_dbfs=False, path = None):

    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 10,            
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    })

    # Étape 1 : Calculer la fréquence moyenne d'activation pour chaque concept
    activations = concept_activations  # (n_samples, n_concepts)
    if activations.ndim == 2:
        frequencies = np.mean(activations, axis=0)  # (n_concepts,)
    else:
        raise ValueError("concept_activations should be 2D (n_samples, n_concepts)")

    concept_names = np.array([c.replace("concept_", "") for c in concept_list])
    leakage_keys = {k.replace('concept_', ''): v for k, v in interconcept_leakage.items()}

    # Calcul du leakage total par concept
    inter_leak_l2 = []
    for concept in concept_names:
        leaks = []
        for other in concept_names:
            if concept != other:
                key_fwd = f"{concept} -> {other}"
                key_bwd = f"{other} -> {concept}"
                leaks.append(leakage_keys.get(key_fwd, leakage_keys.get(key_bwd, 0.0)))
        leaks = np.array(leaks)
        inter_leak_l2.append(np.linalg.norm(leaks))
    inter_leak_l2 = np.array(inter_leak_l2)

    acc_values = np.array([per_concept_acc[f"concept_{c}"] for c in concept_names])

    leakage_low_all = inter_leak_l2[(frequencies >= 0.0) & (frequencies < 0.8)]
    leakage_high_all = inter_leak_l2[(frequencies >= 0.8) & (frequencies <= 1.0)]

    t_stat1, p_val1 = ttest_ind(leakage_low_all, leakage_high_all, equal_var=False)

    fig1, axes1 = plt.subplots(1, 2, figsize=(18, 6))
    axes1[0].boxplot(
        [leakage_low_all, leakage_high_all],
        labels=["Activation ∈ [0, 0.8)", "Activation ∈ [0.8, 1]"],
        showfliers=True,
        patch_artist=True,
        boxprops=dict(facecolor="lightblue"),
        medianprops=dict(color="black")
    )
    axes1[0].set_ylabel("Interconcept leakage score", fontsize=14)
    axes1[0].grid(axis="y", linestyle="--", alpha=0.6)
    axes1[0].text(
        0.98, 0.95,
        f"t = {t_stat1:.2f}\np = {p_val1:.4f}",
        transform=axes1[0].transAxes,
        ha="right", va="top",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )
    axes1[0].tick_params(axis="x", labelsize=13)

    ## Extremums d'accuracy
    sorted_indices = np.argsort(acc_values)
    worst_indices = sorted_indices[:20]
    best_indices = sorted_indices[-20:]
    selected_indices = np.concatenate([worst_indices, best_indices])

    selected_frequencies = frequencies[selected_indices]
    selected_leakage = inter_leak_l2[selected_indices]

    low_activation_mask = (selected_frequencies >= 0.0) & (selected_frequencies < 0.8)
    high_activation_mask = (selected_frequencies >= 0.8) & (selected_frequencies <= 1.0)

    leakage_low = selected_leakage[low_activation_mask]
    leakage_high = selected_leakage[high_activation_mask]

    t_stat2, p_val2 = ttest_ind(leakage_low, leakage_high, equal_var=False)

    axes1[1].boxplot(
        [leakage_low, leakage_high],
        labels=["Activation ∈ [0, 0.8)", "Activation ∈ [0.8, 1]"],
        showfliers=True,
        patch_artist=True,
        boxprops=dict(facecolor="lightgray"),
        medianprops=dict(color="black")
    )

    axes1[1].set_ylabel("Interconcept leakage score", fontsize=14)
    axes1[1].grid(axis="y", linestyle="--", alpha=0.6)
    axes1[1].text(
        0.98, 0.95,
        f"t = {t_stat2:.2f}\np = {p_val2:.4f}",
        transform=axes1[1].transAxes,
        ha="right", va="top",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )
    axes1[1].tick_params(axis="x", labelsize=13)

    if save_to_dbfs:
        save_fig_to_dbfs(fig1,filename="interconcept_leakage_boxplot.png",base_path=path)

    plt.show()

    # --- FIGURE 2 : Task-Concept Leakage ---
    task_leak_values = np.array([task_concept_leakage[f"concept_{c}"] for c in concept_names])

    task_leak_low_all = task_leak_values[(frequencies >= 0.0) & (frequencies < 0.8)]
    task_leak_high_all = task_leak_values[(frequencies >= 0.8) & (frequencies <= 1.0)]

    t_stat3, p_val3 = ttest_ind(task_leak_low_all, task_leak_high_all, equal_var=False)

    fig2, axes2 = plt.subplots(1, 2, figsize=(18, 6))
    axes2[0].boxplot(
        [task_leak_low_all, task_leak_high_all],
        labels=["Activation ∈ [0, 0.8)", "Activation ∈ [0.8, 1]"],
        showfliers=True,
        widths=0.6,
        patch_artist=False,  
        boxprops=dict(linewidth=0.8),
        medianprops=dict(linewidth=1.2),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
    )

    axes2[0].set_ylabel("Task leakage score")
    axes2[0].grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    axes2[0].text(
        0.97, 0.95,
        f"$t = {t_stat3:.2f}$\n$p = {p_val3:.2e}$",
        transform=axes2[0].transAxes,
        ha="right", va="top",
        fontsize=9,
    )

    axes2[0].spines["top"].set_visible(False)
    axes2[0].spines["right"].set_visible(False)

    selected_task_leak = task_leak_values[selected_indices]

    low_activation_mask = (selected_frequencies >= 0.0) & (selected_frequencies < 0.8)
    high_activation_mask = (selected_frequencies >= 0.8) & (selected_frequencies <= 1.0)

    task_leak_low = selected_task_leak[low_activation_mask]
    task_leak_high = selected_task_leak[high_activation_mask]

    t_stat4, p_val4 = ttest_ind(task_leak_low, task_leak_high, equal_var=False)

    axes2[1].boxplot(
        [task_leak_low, task_leak_high],
        labels=["Activation ∈ [0, 0.8)", "Activation ∈ [0.8, 1]"],
        showfliers=True,
        widths=0.6,
        patch_artist=False,
        boxprops=dict(linewidth=0.8),
        medianprops=dict(linewidth=1.2),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
    )

    axes2[1].set_ylabel("Task leakage score")
    axes2[1].grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)

    axes2[1].text(
        0.97, 0.95,
        f"$t = {t_stat4:.2f}$\n$p = {p_val4:.2e}$",
        transform=axes2[1].transAxes,
        ha="right", va="top",
        fontsize=9,
    )

    axes2[1].spines["top"].set_visible(False)
    axes2[1].spines["right"].set_visible(False)

    if save_to_dbfs:
        save_fig_to_dbfs(fig2,filename="task_leakage_boxplot.png",base_path=path)

    plt.show()

    ## box plot by accuracy
    acc_th = np.percentile(acc_values, 8)
    high_acc_mask = acc_values <= acc_th
    low_acc_mask = acc_values > acc_th
    

    inter_leak_low_acc = inter_leak_l2[low_acc_mask]
    inter_leak_high_acc = inter_leak_l2[high_acc_mask]

    t_stat5, p_val5 = ttest_ind(inter_leak_low_acc, inter_leak_high_acc,equal_var=False)

    fig3, axes3 = plt.subplots(1, 2, figsize=(18, 6))
    axes3[0].boxplot(
        [inter_leak_low_acc, inter_leak_high_acc],
        labels=["bottom 85%", "top 15%"],
        showfliers=True,
        patch_artist=True,
        boxprops=dict(facecolor="lightgreen"),
        medianprops=dict(color="black")
    )
    axes3[0].set_ylabel("Interconcept leakage score", fontsize=14)
    axes3[0].grid(axis="y", linestyle="--", alpha=0.6)
    axes3[0].text(
        0.98, 0.95,
        f"t = {t_stat5:.2f}\np = {p_val5:.4f}",
        transform=axes3[0].transAxes,
        ha="right", va="top",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )
    axes3[0].tick_params(axis="x", labelsize=13)

    task_leak_low_acc = task_leak_values[low_acc_mask]
    task_leak_high_acc = task_leak_values[high_acc_mask]

    t_stat6, p_val6 = ttest_ind(task_leak_low_acc,task_leak_high_acc,equal_var=False)

    axes3[1].boxplot(
        [task_leak_low_acc, task_leak_high_acc],
        labels=["bottom 85%", "top 15%"],
        showfliers=True,
        patch_artist=True,
        boxprops=dict(facecolor="lightgreen"),
        medianprops=dict(color="black")
    )
    axes3[1].set_ylabel("Task leakage score", fontsize=14)
    axes3[1].grid(axis="y", linestyle="--", alpha=0.6)
    axes3[1].text(
        0.98, 0.95,
        f"t = {t_stat6:.2f}\np = {p_val6:.4f}",
        transform=axes3[1].transAxes,
        ha="right", va="top",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )
    axes3[1].tick_params(axis="x", labelsize=13)

    if save_to_dbfs:
        save_fig_to_dbfs(fig3,filename="leakage_by_accuracy_boxplot.png",base_path=path)

    plt.show()

def plot_leakage_graph_by_activation(concept_activations, interconcept_leakage, concept_list, per_concept_acc, task_concept_leakage, num_top_concepts=6):

    # Étape 1 : Calculer la fréquence moyenne d'activation pour chaque concept
    activations = concept_activations  # (n_samples, n_concepts)
    if activations.ndim == 2:
        frequencies = np.mean(activations, axis=0)  # (n_concepts,)
    else:
        raise ValueError("concept_activations should be 2D (n_samples, n_concepts)")

    concept_names = np.array([c.replace("concept_", "") for c in concept_list])
    leakage_keys = {k.replace('concept_', ''): v for k, v in interconcept_leakage.items()}

    # Calcul du leakage total par concept
    leakage_sums = []
    for concept in concept_names:
        total_leak = 0
        for other in concept_names:
            if concept != other:
                key_fwd = f"{concept} -> {other}"
                key_bwd = f"{other} -> {concept}"
                total_leak += leakage_keys.get(key_fwd, leakage_keys.get(key_bwd, 0))
        leakage_sums.append(total_leak)
    leakage_sums = np.array(leakage_sums)

    # observation des concepts les moins bien detectés / mieux detectés
    acc_values = np.array([per_concept_acc[f"concept_{c}"] for c in concept_names])

    sorted_indices = np.argsort(acc_values)
    worst_indices = sorted_indices[:20]
    best_indices = sorted_indices[-20:]

    # --- FIGURE 1 : Leakage inter-concept ---
    plt.figure(figsize=(15, 12))

    plt.subplot(2, 1, 1)
    plt.scatter(frequencies[worst_indices], leakage_sums[worst_indices], s=100, c="green", edgecolors="black", label="15 most accurate concepts")
    plt.scatter(frequencies[best_indices], leakage_sums[best_indices], s=100, c="orange", edgecolors="black", label="15 less accurate concepts")

    for i in worst_indices:
        plt.text(frequencies[i] + 0.005, leakage_sums[i], concept_names[i], fontsize=9, color="darkgreen")
    for i in best_indices:
        plt.text(frequencies[i] + 0.005, leakage_sums[i], concept_names[i], fontsize=9, color="darkorange")

    plt.xlabel("Activation frequency")
    plt.ylabel("Total leakage score")
    plt.title("Leakage score by Concept Activation")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    # --- FIGURE 2 : Task-Concept Leakage ---
    task_leak_values = np.array([task_concept_leakage[f"concept_{c}"] for c in concept_names])

    plt.subplot(2, 1, 2)
    plt.scatter(frequencies[worst_indices], task_leak_values[worst_indices], s=100, c="green", edgecolors="black", label="15 most accurate concepts")
    plt.scatter(frequencies[best_indices], task_leak_values[best_indices], s=100, c="orange", edgecolors="black", label="15 less accurate concepts")

    for i in worst_indices:
        plt.text(frequencies[i] + 0.005, task_leak_values[i], concept_names[i], fontsize=9, color="darkgreen")
    for i in best_indices:
        plt.text(frequencies[i] + 0.005, task_leak_values[i], concept_names[i], fontsize=9, color="darkorange")

    plt.xlabel("Activation frequency")
    plt.ylabel("Task leakage score")
    plt.title("Task-Concept Leakage")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_leakage_graph_by_cosine_similarity(concept_list, interconcept_leakage, model):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = model.processor
    clip_model = model.base_model
    model.eval()

    concept_names = [str(c).replace("concept_", "").strip() for c in concept_list]
    n_concepts = len(concept_list)
    inputs = processor(text=concept_names, return_tensors="pt", padding=True, truncation=True)

    for k, v in inputs.items():
        inputs[k] = v.to(device)

    with torch.no_grad():
        text_feats = clip_model.get_text_features(**inputs)
    
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    clip_embeddings = text_feats.cpu().numpy()

    cosine_similarity_matrix = cosine_similarity(clip_embeddings)
    

    leakage_keys = {}

    for k, v in interconcept_leakage.items():
        parts = re.split(r'\s*->\s*', k)
        if len(parts) == 2:
            a, b = parts
            a = a.replace("concept_", "").strip()
            b = b.replace("concept_", "").strip()
            leakage_keys[f"{a}->{b}"] = v
    
    xs, ys, labels = [], [], []
    for i in range(n_concepts):
        for j in range(i+1, n_concepts):
            a, b = concept_names[i], concept_names[j]
            sim = float(cosine_similarity_matrix[i, j])
            leakage = leakage_keys.get(f"{a}->{b}", leakage_keys.get(f"{b}->{a}", 0.0))
            xs.append(sim)
            ys.append(leakage)
            labels.append(f"{a} ↔ {b}")

    xs = np.array(xs)
    ys = np.array(ys)

    plt.figure(figsize=(14, 6))
    plt.scatter(xs, ys, s=30, facecolors="lightblue", edgecolors="black", alpha=0.8)

    plt.xlabel("Cosine similarity (CLIP)")
    plt.ylabel("Leakage score")
    plt.title(f"Cosine similarity vs Leakage — {len(xs)} paires")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

def plot_leakage_graph_by_concept_weight_similarity(concept_list, interconcept_leakage, model, save_to_dbfs = False, path = None):
    
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    concept_names = [str(c).replace("concept_", "").strip() for c in concept_list]
    n_concepts = len(concept_list)
    
    if hasattr(model, "concept_layer"):
        concept_weight_matrix = model.concept_layer.weight.detach().cpu().numpy()
        norm_weights = concept_weight_matrix / np.linalg.norm(
            concept_weight_matrix, axis=1, keepdims=True
        )
    else:
        concept_weight_matrix_text = model.concept_layer_text.weight.detach().cpu().numpy()
        norm_weights_text = concept_weight_matrix_text / np.linalg.norm(
            concept_weight_matrix_text, axis=1, keepdims=True
        )
        concept_weight_matrix_image = model.concept_layer_image.weight.detach().cpu().numpy()
        norm_weights_image = concept_weight_matrix_image / np.linalg.norm(
            concept_weight_matrix_image, axis=1, keepdims=True
        )
    
    cosine_similarity_matrix = cosine_similarity(norm_weights)

    leakage_keys = {}
    for k, v in interconcept_leakage.items():
        parts = re.split(r'\s*->\s*', k)
        if len(parts) == 2:
            a, b = parts
            a = a.replace("concept_", "").strip()
            b = b.replace("concept_", "").strip()
            leakage_keys[f"{a}->{b}"] = v
    
    xs, ys, labels = [], [], []
    for i in range(n_concepts):
        for j in range(i+1, n_concepts):
            a, b = concept_names[i], concept_names[j]
            sim = float(cosine_similarity_matrix[i, j])
            leakage = leakage_keys.get(f"{a}->{b}", leakage_keys.get(f"{b}->{a}", 0.0))
            xs.append(sim)
            ys.append(leakage)
            labels.append(f"{a} ↔ {b}")

    fig, ax = plt.subplots(figsize=(14, 6))

    xs = np.array(xs)
    ys = np.array(ys)

    ax.scatter(xs,ys,s=18,facecolors="lightblue",edgecolors="black",linewidth=0.6,alpha=0.9)
    coef = np.polyfit(xs, ys, 2)
    x_fit = np.linspace(xs.min(), xs.max(), 200)
    y_fit = np.polyval(coef, x_fit)

    ax.plot(x_fit,y_fit,color="red",linewidth=1.2)

    r, p = pearsonr(xs, ys)

    ax.text(
        0.03, 0.95,
        f"$r = {r:.2f}$\n$p = {p:.2e}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )

    ax.set_xlabel("Weight similarity (Fully connected − CBL)")
    ax.set_ylabel("Leakage score")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    plt.show()

    if save_to_dbfs:
        save_fig_to_dbfs(fig,filename="weight_similarity.png",base_path=path)

# def plot_leakage_graph_by_concept_weight_similarity(concept_list, interconcept_leakage, model):
    
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.eval()

#     concept_text = [c for c in concept_list if "text" in str(c).lower()]
#     concept_image = [c for c in concept_list if "image" in str(c).lower()]

#     # Pour enlever "concept_", "text_", "image_" etc.
#     def clean_name(c):
#         return str(c).replace("concept_", "").replace("text_", "").replace("image_", "").strip()

#     concept_names_text = [clean_name(c) for c in concept_text]
#     concept_names_image = [clean_name(c) for c in concept_image]

#     if hasattr(model, "concept_layer"):
#         raise ValueError(
#             "Ton modèle n’a qu’une seule couche concept_layer : impossible de séparer texte/image."
#         )

#     # Poids text
#     w_text = model.concept_layer_text.weight.detach().cpu().numpy()
#     norm_text = w_text / np.linalg.norm(w_text, axis=1, keepdims=True)

#     # Poids image
#     w_image = model.concept_layer_image.weight.detach().cpu().numpy()
#     norm_image = w_image / np.linalg.norm(w_image, axis=1, keepdims=True)

#     sim_text = cosine_similarity(norm_text)
#     sim_image = cosine_similarity(norm_image)

#     leakage_keys = {}
#     for k, v in interconcept_leakage.items():
#         parts = re.split(r'\s*->\s*', k)
#         if len(parts) == 2:
#             a, b = parts
#             a = clean_name(a)
#             b = clean_name(b)
#             leakage_keys[f"{a}->{b}"] = v

#     def build_xy(concept_names, similarity_matrix):
#         xs, ys, labels = [], [], []
#         n = len(concept_names)

#         for i in range(n):
#             for j in range(i+1, n):
#                 a, b = concept_names[i], concept_names[j]
#                 sim = float(similarity_matrix[i, j])
#                 leakage = leakage_keys.get(f"{a}->{b}", leakage_keys.get(f"{b}->{a}", 0.0))
                
#                 xs.append(sim)
#                 ys.append(leakage)
#                 labels.append(f"{a} ↔ {b}")

#         return np.array(xs), np.array(ys), labels

#     xs_text, ys_text, labels_text = build_xy(concept_names_text, sim_text)
#     xs_image, ys_image, labels_image = build_xy(concept_names_image, sim_image)

#     plt.figure(figsize=(14, 6))
#     plt.scatter(xs_text, ys_text, s=30, facecolors="lightblue", edgecolors="black", alpha=0.8)
#     plt.xlabel("Weight similarity (TEXT)")
#     plt.ylabel("Leakage score")
#     plt.title(f"Text: Similarité des poids vs Leakage — {len(xs_text)} paires")
#     plt.grid(True, linestyle="--", alpha=0.4)
#     plt.tight_layout()
#     plt.show()

#     plt.figure(figsize=(14, 6))
#     plt.scatter(xs_image, ys_image, s=30, facecolors="salmon", edgecolors="black", alpha=0.8)
#     plt.xlabel("Weight similarity (IMAGE)")
#     plt.ylabel("Leakage score")
#     plt.title(f"Image: Similarité des poids vs Leakage — {len(xs_image)} paires")
#     plt.grid(True, linestyle="--", alpha=0.4)
#     plt.tight_layout()
#     plt.show()

def plot_leakage_graph_by_concept_co_activation(concept_activations, concept_list, interconcept_leakage, model):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    concept_names = [str(c).replace("concept_", "").strip() for c in concept_list]
    n_concepts = len(concept_list)
    
    assert concept_activations.shape[1] == n_concepts
    norm_activations = concept_activations / np.linalg.norm(concept_activations, axis=1, keepdims=True)
    
    cosine_similarity_matrix = cosine_similarity(norm_activations)

    leakage_keys = {}
    for k, v in interconcept_leakage.items():
        parts = re.split(r'\s*->\s*', k)
        if len(parts) == 2:
            a, b = parts
            a = a.replace("concept_", "").strip()
            b = b.replace("concept_", "").strip()
            leakage_keys[f"{a}->{b}"] = v
    
    xs, ys, labels = [], [], []
    for i in range(n_concepts):
        for j in range(i+1, n_concepts):
            a, b = concept_names[i], concept_names[j]
            sim = float(cosine_similarity_matrix[i, j])
            leakage = leakage_keys.get(f"{a}->{b}", leakage_keys.get(f"{b}->{a}", 0.0))
            xs.append(sim)
            ys.append(leakage)
            labels.append(f"{a} ↔ {b}")

    xs = np.array(xs)
    ys = np.array(ys)

    plt.figure(figsize=(14, 6))
    plt.scatter(xs, ys, s=30, facecolors="lightblue", edgecolors="black", alpha=0.8)

    plt.xlabel("Cosine similarity of concept activations")
    plt.ylabel("Leakage score")
    plt.title(f"co-activation vs Leakage — {len(xs)} paires")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()



def plot_leakage_graph_by_semantic_similarity(concept_list, interconcept_leakage):
    """
    Plot leakage score vs semantic similarity provided from Claude..
    
    Args:
        interconcept_leakage: Dictionary with leakage values for the concept pairs
    """
    semantic_similarities = {
    "machine learning -> artificial intelligence": 0.98,
    "team sports -> individual athletics": 0.95,
    "virtual reality -> augmented reality": 0.94,
    "data privacy -> cybersecurity": 0.93,
    "music festivals -> concert tours": 0.92,
    "streaming platforms -> distribution networks": 0.91,
    "cooking techniques -> preparation methods": 0.91,
    "chef profiles -> artist interviews": 0.90,
    "recipe guides -> cooking techniques": 0.89,
    "record labels -> music publishers": 0.89,
    "regulatory bodies -> regulatory agencies": 0.88,
    "agricultural production -> farmers associations": 0.88,
    "athlete profiles -> player transfers": 0.87,
    "championships -> tournaments": 0.87,
    "cloud computing -> Internet of Things": 0.86,
    "food science -> nutrition research": 0.86,
    "streaming metrics -> chart rankings": 0.85,
    "biotechnology -> healthcare technology": 0.84,
    "social media -> internet platforms": 0.83,
    "APIs -> programming frameworks": 0.83,
    "quantum computing -> cuisine trends": 0.02,
    "space technology -> recipe guides": 0.03,
    "cryptocurrency -> chef profiles": 0.04,
    "robotics -> food festivals": 0.05,
    "biotechnology -> esports": 0.06,
    "machine learning -> terroir": 0.07,
    "cybersecurity -> dietary patterns": 0.08,
    "cloud computing -> fermentation": 0.09,
    "quantum computing -> chef interviews": 0.09,
    "APIs -> flavor profiles": 0.10,
    "data analytics -> culinary awards": 0.11,
    "artificial intelligence -> restaurant reviews": 0.12,
    "Internet of Things -> ingredients": 0.13,
    "programming frameworks -> tasting evaluations": 0.14,
    "venture capital -> seasonal availability": 0.15,
    "research institutions -> farmers associations": 0.16,
    "clean energy tech -> comfort food preferences": 0.17,
    "startups -> gastronomic techniques": 0.18,
    "technical standards -> preservation processes": 0.19,
    "networking infrastructure -> market pricing": 0.20
    }
    
    xs, ys, labels = [], [], []
    leakage_keys = {}
    for k, v in interconcept_leakage.items():
        parts = re.split(r'\s*->\s*', k)
        if len(parts) == 2:
            a, b = parts
            a = a.replace("concept_", "").strip()
            b = b.replace("concept_", "").strip()
            leakage_keys[f"{a}->{b}"] = v
    
    for pair_key, sim_score in semantic_similarities.items():
        # Extract concept names from the pair key
        parts = re.split(r'\s*->\s*', pair_key)
        if len(parts) == 2:
            a, b = parts
            a = a.strip()
            b = b.strip()
            
            # Try to find leakage value in interconcept_leakage
            leakage = leakage_keys.get(f"{a}->{b}", leakage_keys.get(f"{b}->{a}", 0.0))
            
            xs.append(sim_score)
            ys.append(leakage)
            labels.append(pair_key)
    
    xs = np.array(xs)
    ys = np.array(ys)
    
    # Create the plot
    plt.figure(figsize=(14, 6))
    plt.scatter(xs, ys, s=50, facecolors="lightblue", edgecolors="black", alpha=0.8)
    
    plt.xlabel("Semantic Similarity Score", fontsize=12)
    plt.ylabel("Leakage Score", fontsize=12)
    plt.title(f"Semantic Similarity vs Leakage — {len(xs)} paires", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_leakage_score_by_co_activation(concept_activations, interconcept_leakage, concept_list):
    """
    interconcept_leakage  : dict {"concept_a->concept_b": leakage score, ...}
    """

    coactivation_matrix = cosine_similarity(concept_activations)

    concept_names = [str(c).replace("concept_", "").strip() for c in concept_list]
    n_concepts = len(concept_names)

    leakage_keys = {}
    for k, v in interconcept_leakage.items():
        parts = re.split(r'\s*->\s*', k)
        if len(parts) == 2:
            a, b = parts
            a = a.replace("concept_", "").strip()
            b = b.replace("concept_", "").strip()
            leakage_keys[f"{a}->{b}"] = v

    xs, ys, labels = [], [], []

    for i in range(n_concepts):
        for j in range(i+1, n_concepts):
            a, b = concept_names[i], concept_names[j]
            coact = float(coactivation_matrix[i, j])
            leakage = leakage_keys.get(f"{a}->{b}", leakage_keys.get(f"{b}->{a}", 0.0))
            xs.append(coact)
            ys.append(leakage)
            labels.append(f"{a} ↔ {b}")

    xs = np.array(xs)
    ys = np.array(ys)

    # Scatter plot (points petits et sans labels)
    plt.figure(figsize=(14, 6))
    plt.scatter(xs, ys, s=30, facecolors="lightblue", edgecolors="black")

    plt.xlabel("Concept co-activation")
    plt.ylabel("Leakage score")
    plt.title("Leakage score by co-activation")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

def heatmap_leakage_score_by_co_activation(concept_activations, interconcept_leakage, concept_list):

    coactivation_matrix = cosine_similarity(concept_activations)

    concept_names = [str(c).replace("concept_", "").strip() for c in concept_list]
    n_concepts = len(concept_names)

    leakage_matrix = np.zeros((n_concepts, n_concepts))

    for k, v in interconcept_leakage.items():
        parts = re.split(r'\s*->\s*', k)
        if len(parts) == 2:
            a, b = parts
            a = a.replace("concept_", "").strip()
            b = b.replace("concept_", "").strip()
            if a in concept_names and b in concept_names:
                i, j = concept_names.index(a), concept_names.index(b)
                leakage_matrix[i, j] = v
    
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        leakage_matrix,
        xticklabels=concept_names,
        yticklabels=concept_names,
        cmap="Reds",
        annot=False,
        fmt=".2f",
        cbar_kws={"label": "Leakage score"}
    )

    ax.set_title("Inter-concept leakage heatmap", fontsize=14, pad=12)
    ax.set_xlabel("To concept", fontsize=12)
    ax.set_ylabel("From concept", fontsize=12)

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_task_vs_interconcept_leakage(interconcept_leakage,task_concept_leakage,concept_list,per_concept_acc=None, save_to_dbfs = False, path = None):
        """
        Correlation between inter-concept leakage (L2 norm)
        and task-concept leakage.
        """

        mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    })

        concept_names = np.array([c.replace("concept_", "") for c in concept_list])

        leakage_keys = {
            k.replace("concept_", ""): v for k, v in interconcept_leakage.items()
        }

        # Interconcept Leakage L2-norm
        inter_leak_l2 = []

        for concept in concept_names:
            leaks = []
            for other in concept_names:
                if concept != other:
                    key_fwd = f"{concept} -> {other}"
                    key_bwd = f"{other} -> {concept}"
                    leaks.append(
                        leakage_keys.get(key_fwd, leakage_keys.get(key_bwd, 0.0))
                    )

            leaks = np.array(leaks)
            inter_leak_l2.append(np.linalg.norm(leaks))

        inter_leak_l2 = np.array(inter_leak_l2)

        #Task Leakage
        task_leak = np.array([
            task_concept_leakage[f"concept_{c}"] for c in concept_names
        ])

        pearson_r, pearson_p = pearsonr(inter_leak_l2, task_leak)
        spearman_r, spearman_p = spearmanr(inter_leak_l2, task_leak)

        fig, ax = plt.subplots()

        if per_concept_acc is not None:
            acc = np.array([per_concept_acc[f"concept_{c}"] for c in concept_names])
            sc = ax.scatter(
                inter_leak_l2,
                task_leak,
                c=acc,
                cmap="viridis",
                s=40,                # smaller for conference style
                edgecolors="black",
                linewidth=0.5
            )
            cbar = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.05)
            cbar.set_label("Concept accuracy", fontsize=9)
            cbar.ax.tick_params(labelsize=8)
        else:
            ax.scatter(
                inter_leak_l2,
                task_leak,
                s=40,
                facecolors="blue",
                edgecolors="black",
                linewidth=0.5
            )

        idx = np.argsort(inter_leak_l2 + task_leak)

        ax.set_xlabel("Inter-concept leakage (L2 norm)")
        ax.set_ylabel("Task-concept leakage")
        ax.text(
            0.03, 0.97,
            (
                f"$r = {pearson_r:.3f}$"
                f"\n$\\rho = {spearman_r:.3f}$"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )

        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()

        if save_to_dbfs:
            save_fig_to_dbfs(fig,filename="inter_vs_task_leakage_scatter.png",base_path=path)

        plt.show()


