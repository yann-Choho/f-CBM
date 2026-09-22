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

class BaselineModel(nn.Module):
    def __init__(self, 
                 embedder_model,
                 classifier, 
                 train_loader, 
                 val_loader = None, 
                 test_loader = None, 
                 config = None, 
                 save_path = None,
                ):
        super().__init__()  # Appel au constructeur de nn.Module

        # Mode image seulement - pas de features multimodales
        self.image_only_features = True
        
        # Ici, nous faisons une copie profonde pour éviter de lier le même objet        
        self.classifier = copy.deepcopy(classifier)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.save_path = save_path
        if config:
            self.device = config.device
        else :
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.num_epochs = self.config.num_epochs
        self.embedder_model_name = self.config.model_name
        self.best_acc_score = 0

        # Initialize encoder and processor based on chosen backbone - IMAGE ONLY
        if config.model_name == 'clip':
            self.embedder_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.tokenizer = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self._is_clip = True
        elif config.model_name == 'blip':
            self.embedder_model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
            self.tokenizer = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self._is_clip = False
        else:
            raise ValueError(f"Unknown backbone: {config.model_name}")

        
        self.embedder_model.to(self.device)
        self.classifier.to(self.device)

        self.optimizer = torch.optim.Adam(list(self.embedder_model.parameters()) + list(self.classifier.parameters()), lr=1e-5)

        # Dictionnaire pour stocker les performances
        self.performances = {
            "train": {},
            "val": {},
            "validation": {},  # Ajout pour compatibilité
            "test": {}
        }

        # pour captum
        self.concat_layer = nn.Identity()
        
    def get_pooled_output(self, pixel_values):
        """
        Return image embedding only.
        For CLIP, use image_embeds; for BLIP, use image features.
        """
        self.embedder_model.to(self.device)
        
        if self._is_clip:
            # CLIP - utilise uniquement les embeddings image
            outputs = self.embedder_model.get_image_features(pixel_values=pixel_values)
            image_features = outputs  # [B, 512] pour CLIP
        else:
            # BLIP - utilise uniquement les features visuelles
            outputs = self.embedder_model.vision_model(pixel_values=pixel_values)
            image_features = outputs.pooler_output  # [B, 768] pour BLIP
    
        return image_features
         
    def forward(self, pixel_values):
        """
        Effectue une passe avant à travers le modèle pour obtenir les prédictions de labels.
        
        Arguments:
        - pixel_values (torch.Tensor): Les tenseurs d'images preprocessées.

        Retourne:
        - torch.Tensor: Les labels prédits pour les images.
        """
        self.embedder_model.eval()  # Désactiver le dropout pour l'inférence
        self.classifier.eval()

        with torch.no_grad():
            # Passe à travers le modèle d'embedding image seulement
            pooled_output = self.get_pooled_output(pixel_values)

            pooled_output   = self.concat_layer(pooled_output)       # rien ne change, mais Captum la “voit”

            # Passer les caractéristiques intégrées au classificateur
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
                # Image seulement - pas de text/input_ids
                images = batch["image"].to(self.device)
                label = batch["label"].to(self.device)

                self.optimizer.zero_grad()
                
                # Get image embeddings only
                pooled_output = self.get_pooled_output(images)
                logits = self.classifier(pooled_output)
                loss = self.loss_fn(logits, label)
                
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                total_predictions += label.size(0)
                correct_predictions += (predicted == label).sum().item()

                # Libérer la mémoire GPU après chaque batch
                del images, label, pooled_output, logits
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

    def evaluate_model(self, dataloader, dataset_name):
        self.embedder_model.eval()
        self.classifier.eval()
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {dataset_name}", unit="batch"):
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                
                pooled_output = self.get_pooled_output(images)
                logits = self.classifier(pooled_output)
                
                predictions = torch.argmax(logits, axis=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                del images, labels, pooled_output, logits
                torch.cuda.empty_cache()
                gc.collect()
        
        # Calcul des métriques
        accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        self.performances[dataset_name.lower()]["accuracy"] = accuracy
        self.performances[dataset_name.lower()]["f1_score"] = f1
        
        print(f"{dataset_name} - Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
        
        return accuracy

    def save_model(self):
        """Sauvegarde les poids du modèle"""
        if self.save_path:
            os.makedirs(f"{self.save_path}/blue_checkpoints/{self.config.model_name}/BaselineModel", exist_ok=True)
            
            # Sauvegarde du classificateur
            classifier_path = f"{self.save_path}/blue_checkpoints/{self.config.model_name}/BaselineModel/{self.config.model_name}_classifier_state_dict.pth"
            torch.save(self.classifier.state_dict(), classifier_path)
            
            # Sauvegarde de l'embedder
            embedder_path = f"{self.save_path}/blue_checkpoints/{self.config.model_name}/BaselineModel/{self.config.model_name}_embedder_state_dict.pth"
            torch.save(self.embedder_model.state_dict(), embedder_path)
            
            print(f"Modèle sauvegardé dans {self.save_path}")

    def load_model(self):
        """Charge les poids du modèle"""
        if self.save_path:
            classifier_path = f"{self.save_path}/blue_checkpoints/{self.config.model_name}/BaselineModel/{self.config.model_name}_classifier_state_dict.pth"
            embedder_path = f"{self.save_path}/blue_checkpoints/{self.config.model_name}/BaselineModel/{self.config.model_name}_embedder_state_dict.pth"
            
            if os.path.exists(classifier_path):
                self.classifier.load_state_dict(torch.load(classifier_path, map_location=self.device))
                print(f"Classificateur chargé depuis {classifier_path}")
            
            if os.path.exists(embedder_path):
                self.embedder_model.load_state_dict(torch.load(embedder_path, map_location=self.device))
                print(f"Embedder chargé depuis {embedder_path}")

    def save_performance_json(self):
        """Sauvegarde les performances dans un fichier JSON"""
        if self.save_path:
            perf_path = f"{self.save_path}/blue_checkpoints/{self.config.model_name}/BaselineModel/performances.json"
            with open(perf_path, 'w') as f:
                json.dump(self.performances, f, indent=2)
            print(f"Performances sauvegardées dans {perf_path}")