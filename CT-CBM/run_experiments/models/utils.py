from transformers import RobertaTokenizer, RobertaModel, BertTokenizer, \
      BertModel, DebertaTokenizer, DebertaModel, GPT2Model, GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM 
    #   ModernBertModel

from cbm_models import ModelXtoCtoY_function

import os
import torch


# Load the model and tokenizer and bottleneck layer
def load_model_and_tokenizer(config, n_concepts = 4):
    """
    n_concepts : nombre de concepts dans le modèle joint (techniquement juste le nombre de neurones dans ModelXtoCtoY_function)
    
    Pour CLIP/BLIP :
      - model = None (instancié directement dans les BaselineModel_clip*.py)
      - tokenizer = CLIPTokenizerFast / BlipTokenizer (extrait du Processor)
        Cela garantit que tous les modules du pipeline reçoivent un tokenizer fonctionnel.
    """
    if config.model_name == 'roberta-base':
        tokenizer = RobertaTokenizer.from_pretrained(config.model_name)
        model = RobertaModel.from_pretrained(config.model_name)
    elif config.model_name == 'roberta-large':
        tokenizer = RobertaTokenizer.from_pretrained(config.model_name)
        model = RobertaModel.from_pretrained(config.model_name)
    elif config.model_name == 'bert-base-uncased':
        tokenizer = BertTokenizer.from_pretrained(config.model_name)
        model = BertModel.from_pretrained(config.model_name)
    elif config.model_name == 'deberta-base':
        tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
        model = DebertaModel.from_pretrained('microsoft/deberta-base')
    elif config.model_name == 'deberta-large':
        tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-large')
        model = DebertaModel.from_pretrained('microsoft/deberta-large')
   
    elif config.model_name == 'gpt2':
        model = GPT2Model.from_pretrained(config.model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(config.model_name)
        tokenizer.pad_token = tokenizer.eos_token
    elif config.model_name == 'gemma':
        hf_token = os.environ["HF_TOKEN"]
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it", use_auth_token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-2-2b-it",
            device_map={"": 0},  # Tout sur GPU 0
            use_auth_token=hf_token,
        )
        model = model.base_model
    elif config.model_name in ('clip', 'clip-large'):
        # Model = None car instancié dans BaselineModel_clip*.py
        # Tokenizer = CLIPTokenizerFast pour que tout le pipeline fonctionne
        from clip_config import get_clip_checkpoint
        from transformers import CLIPProcessor
        checkpoint = get_clip_checkpoint(config.model_name)
        model = None
        tokenizer = CLIPProcessor.from_pretrained(checkpoint).tokenizer
    elif config.model_name == 'blip':
        # Model = None car instancié dans BaselineModel_image.py / BaselineModel_clip_text.py
        # Tokenizer = BlipTokenizer pour que tout le pipeline fonctionne
        from clip_config import get_clip_checkpoint
        from transformers import BlipProcessor
        checkpoint = get_clip_checkpoint(config.model_name)
        model = None
        tokenizer = BlipProcessor.from_pretrained(checkpoint).tokenizer
    elif config.model_name == 'lstm':
        # Implement the LSTM model setup here
        pass

    # load the bottleneck layer
    if config.model_name == 'lstm':
        ModelXtoCtoY_layer = ModelXtoCtoY_function(
            concept_classes=config.num_each_concept_classes,
            label_classes=config.num_labels,
            n_attributes=n_concepts,
            bottleneck=True,
            expand_dim=config.expand_dim,
            n_class_attr=config.num_each_concept_classes,
            use_relu=False,
            use_sigmoid=False,
            Lstm=True,
            aux_logits=config.is_aux_logits,
            config = config)
    else:
        ModelXtoCtoY_layer = ModelXtoCtoY_function(
            concept_classes= config.num_each_concept_classes,
            label_classes=config.num_labels,
            n_attributes= n_concepts,
            bottleneck=True,
            expand_dim=config.expand_dim,
            n_class_attr=config.num_each_concept_classes,
            use_relu=False,
            use_sigmoid=False,
            aux_logits=config.is_aux_logits,
            config = config)

    # load the linear classifier layer for the baseline

    if config.model_name in ['clip', 'clip-large', 'blip']:
        classifier = nn.Sequential(
            torch.nn.Linear(config.dim, config.num_labels)        
        )
    else:
        if config.use_relu :
            classifier = torch.nn.Sequential(
                torch.nn.Linear(model.config.hidden_size, config.projection),
                nn.Dropout(config.dropout),
                torch.nn.Sigmoid(),
                torch.nn.Linear(config.projection, config.num_labels),        
           )
        elif config.use_relu:
            classifier = torch.nn.Sequential(
                torch.nn.Linear(model.config.hidden_size, config.projection),     
                nn.Dropout(config.dropout),
                torch.nn.ReLU(),
                torch.nn.Linear(config.projection, config.num_labels)       
            )
        else :
            classifier = nn.Sequential(
                torch.nn.Linear(model.config.hidden_size, config.num_labels)
            )

    # model.config.hidden_size is a KEYWORD here : attribute of the model
    return model, tokenizer, ModelXtoCtoY_layer, classifier

# -------------------------- SPARSE LINEAR LAYER -------------

import torch
import torch.nn as nn

class RidgeLinearLayer(nn.Module):
    def __init__(self, input_dim, output_dim, l2_lambda):
        super(RidgeLinearLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.l2_lambda = l2_lambda

    def forward(self, x):
        return self.linear(x)

    def l2_penalty(self):
        return self.l2_lambda * torch.sum(self.linear.weight ** 2)

    def ridge_loss(self, outputs, targets):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, targets) + self.l2_penalty()
        return loss
    
class ElasticNetLinearLayer(nn.Module):
    def __init__(self, in_features, out_features, alpha=0.01, l1_ratio=0.5):
        super(ElasticNetLinearLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        
    def forward(self, x):
        return self.linear(x)
    
    def elasticnet_loss(self, outputs, targets):
        criterion = nn.CrossEntropyLoss()
        l1_norm = torch.norm(self.linear.weight, p=1)
        l2_norm = torch.norm(self.linear.weight, p=2)
        loss = criterion(outputs, targets) + self.alpha * (self.l1_ratio * l1_norm + (1 - self.l1_ratio) * l2_norm)
        return loss
    
    def reset_parameters(self):
        """Réinitialiser les paramètres de la couche linéaire."""
        self.linear.reset_parameters()
    
    def reset_residual_layer(self):
        """Réinitialiser la couche résiduelle (linear_layer)."""
        self.reset_parameters()
