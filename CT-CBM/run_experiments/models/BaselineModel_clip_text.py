import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
import gc
import os
import copy
import json
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor, BlipModel, BlipProcessor
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

class BaselineModel(nn.Module):
    def __init__(self, 
                 embedder_model,
                 classifier, 
                 train_loader, 
                 val_loader=None, 
                 test_loader=None, 
                 config=None, 
                 save_path=None,
                ):
        super().__init__()

        # Mode text seulement - pas de features image
        self.text_only_features = True
        self.multimodal_features = False
        self.image_only_features = False
        
        self.classifier = copy.deepcopy(classifier)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.save_path = save_path
        if config:
            self.device = config.device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.num_epochs = self.config.num_epochs
        self.embedder_model_name = self.config.model_name
        self.best_acc_score = 0

        # Initialize encoder and processor based on chosen backbone - TEXT ONLY
        from clip_config import get_clip_checkpoint, is_clip_model
        checkpoint = get_clip_checkpoint(config.model_name)
        
        if is_clip_model(config.model_name):
            self.embedder_model = CLIPModel.from_pretrained(checkpoint)
            self.tokenizer = CLIPProcessor.from_pretrained(checkpoint)
            self._is_clip = True
        else:  # BLIP
            self.embedder_model = BlipModel.from_pretrained(checkpoint)
            self.tokenizer = BlipProcessor.from_pretrained(checkpoint)
            self._is_clip = False

        self.embedder_model.to(self.device)
        self.classifier.to(self.device)

        self.optimizer = torch.optim.Adam(
            list(self.embedder_model.parameters()) + list(self.classifier.parameters()), 
            lr=1e-5
        )

        # Dictionnaire pour stocker les performances
        self.performances = {
            "train": {},
            "val": {},
            "validation": {},
            "test": {}
        }

        # Pour captum
        self.concat_layer = nn.Identity()
        
    def get_pooled_output(self, input_ids, attention_mask):
        """
        Return text embedding only.
        For CLIP, use get_text_features; for BLIP, use text_model pooler_output.
        """
        self.embedder_model.to(self.device)
        
        if self._is_clip:
            # CLIP - utilise uniquement les embeddings texte
            text_features = self.embedder_model.get_text_features(
                input_ids=input_ids, 
                attention_mask=attention_mask
            )
            return text_features  # [B, 512] pour CLIP
        else:
            # BLIP - utilise uniquement les features textuelles
            outputs = self.embedder_model.text_model(
                input_ids=input_ids, 
                attention_mask=attention_mask
            )
            return outputs.pooler_output  # [B, 768] pour BLIP
         
    def forward(self, input_ids, attention_mask):
        """
        Effectue une passe avant à travers le modèle pour obtenir les prédictions de labels.
        
        Arguments:
        - input_ids (torch.Tensor): Les identifiants des tokens du texte.
        - attention_mask (torch.Tensor): Le masque d'attention.

        Retourne:
        - torch.Tensor: Les labels prédits pour les textes.
        """
        self.embedder_model.eval()
        self.classifier.eval()

        with torch.no_grad():
            # Passe à travers le modèle d'embedding texte seulement
            pooled_output = self.get_pooled_output(input_ids, attention_mask)

            pooled_output = self.concat_layer(pooled_output)  # Pour Captum

            # Passer les caractéristiques au classificateur
            logits = self.classifier(pooled_output)
            
            # Obtenir les prédictions finales
            predictions = torch.argmax(logits, axis=1)
        
        return predictions

    def train_model(self):
        for epoch in range(self.num_epochs):
            self.embedder_model.train()
            self.classifier.train()
            running_loss = 0.0
            correct_predictions = 0
            total_predictions = 0
            
            for batch in tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}", unit="batch"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                label = batch["label"].to(self.device)

                self.optimizer.zero_grad()
                
                # Get text embeddings only
                pooled_output = self.get_pooled_output(input_ids, attention_mask)
                logits = self.classifier(pooled_output)
                loss = self.loss_fn(logits, label)
                
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                total_predictions += label.size(0)
                correct_predictions += (predicted == label).sum().item()

                # Libérer la mémoire GPU après chaque batch
                del input_ids, attention_mask, label, pooled_output, logits
                torch.cuda.empty_cache()
                gc.collect()
            
            epoch_loss = running_loss / len(self.train_loader)
            epoch_acc = correct_predictions / total_predictions
            
            self.performances["train"][f"epoch_{epoch+1}"] = {
                "loss": epoch_loss,
                "accuracy": epoch_acc
            }
            
            print(f"Epoch {epoch+1}/{self.num_epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
            
            # Validation
            if self.val_loader:
                val_acc = self.evaluate_model(self.val_loader, 'Validation')
                if val_acc > self.best_acc_score:
                    self.best_acc_score = val_acc
                    self.save_model()
        
        if self.val_loader is None:
            self.save_model()

    def evaluate_model(self, dataloader, dataset_name):
        self.embedder_model.eval()
        self.classifier.eval()
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {dataset_name}", unit="batch"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)
                
                pooled_output = self.get_pooled_output(input_ids, attention_mask)
                logits = self.classifier(pooled_output)
                
                predictions = torch.argmax(logits, axis=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                del input_ids, attention_mask, labels, pooled_output, logits
                torch.cuda.empty_cache()
                gc.collect()
        
        # Calcul des métriques
        accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        self.performances[dataset_name.lower()]["accuracy"] = accuracy
        self.performances[dataset_name.lower()]["f1_score"] = f1
        
        print(f"{dataset_name} - Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
        
        return accuracy

    def _get_state_suffix(self):
        """Get suffix for sigmoid/relu variant (aligned with dispatcher checkpoint path)."""
        if hasattr(self.config, 'sigmoid_or_relu_state') and self.config.sigmoid_or_relu_state != 'linearity':
            return f"_{self.config.sigmoid_or_relu_state}"
        return ""

    def _get_base_dir(self):
        """Get base directory for model checkpoints."""
        return f"{self.save_path}/blue_checkpoints/{self.config.model_name}/BaselineModel"

    def save_model(self):
        """Sauvegarde les poids du modèle"""
        if self.save_path:
            base_dir = self._get_base_dir()
            os.makedirs(base_dir, exist_ok=True)
            suffix = self._get_state_suffix()
            
            # Sauvegarde du classificateur (avec _text_ pour distinguer de image/multimodal)
            classifier_path = f"{base_dir}/{self.config.model_name}_text_classifier_state_dict{suffix}.pth"
            torch.save(self.classifier.state_dict(), classifier_path)
            
            # Sauvegarde de l'embedder
            embedder_path = f"{base_dir}/{self.config.model_name}_text_embedder_state_dict{suffix}.pth"
            torch.save(self.embedder_model.state_dict(), embedder_path)
            
            print(f"Modèle sauvegardé : {classifier_path}")

    def load_model(self):
        """Charge les poids du modèle"""
        if self.save_path:
            base_dir = self._get_base_dir()
            suffix = self._get_state_suffix()
            
            classifier_path = f"{base_dir}/{self.config.model_name}_text_classifier_state_dict{suffix}.pth"
            embedder_path = f"{base_dir}/{self.config.model_name}_text_embedder_state_dict{suffix}.pth"
            
            if os.path.exists(classifier_path):
                self.classifier.load_state_dict(torch.load(classifier_path, map_location=self.device))
                print(f"Classificateur chargé depuis {classifier_path}")
            else:
                print(f"⚠️ Classifier checkpoint non trouvé : {classifier_path}")
            
            if os.path.exists(embedder_path):
                self.embedder_model.load_state_dict(torch.load(embedder_path, map_location=self.device))
                print(f"Embedder chargé depuis {embedder_path}")
            else:
                print(f"⚠️ Embedder checkpoint non trouvé : {embedder_path}")
            
            self.embedder_model.to(self.device)
            self.classifier.to(self.device)
            self.embedder_model.eval()
            self.classifier.eval()
            
            # Charger les performances si le fichier existe
            perf_path = f"{base_dir}/text_performances{suffix}.json"
            if os.path.exists(perf_path):
                with open(perf_path, "r") as f:
                    self.performances = json.load(f)
                print(f"Performances chargées depuis {perf_path}")
            
            gc.collect()
            torch.cuda.empty_cache()

    def save_performance_json(self):
        """Sauvegarde les performances dans un fichier JSON"""
        if self.save_path:
            base_dir = self._get_base_dir()
            os.makedirs(base_dir, exist_ok=True)
            suffix = self._get_state_suffix()
            perf_path = f"{base_dir}/text_performances{suffix}.json"
            with open(perf_path, 'w') as f:
                json.dump(self.performances, f, indent=2)
            print(f"Performances sauvegardées dans {perf_path}")
