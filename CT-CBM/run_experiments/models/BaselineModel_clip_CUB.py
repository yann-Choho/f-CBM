import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
import gc
import os
import sys
import copy
import json
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor, BlipModel, BlipProcessor

# clip_config.py lives in scripts/ — make it importable from models/
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))


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
        super().__init__()

        self.multimodal_features = False
        
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

        # Initialize encoder and processor from centralized registry
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

        # CHANGE 1: Replaced single Adam optimizer (lr=1e-5) with differentiated AdamW param groups:
        # backbone at lr=1e-5, classifier at lr=1e-1 (matching CustomClassifier aggressive training)
        clip_params = [p for p in self.embedder_model.parameters() if p.requires_grad]
        classifier_params = [p for p in self.classifier.parameters() if p.requires_grad]
        param_groups = []
        if len(clip_params) > 0:
            param_groups.append({'params': clip_params, 'lr': 1e-5})
        if classifier_params:
            param_groups.append({'params': classifier_params, 'lr': 1e-1})
        self.optimizer = torch.optim.AdamW(param_groups)

        # CHANGE 2: Added cosine annealing scheduler (eta_min=1e-5, matching CustomClassifier)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.num_epochs - 1, eta_min=1e-5
        )

        self.performances = {
            "train": {},
            "val": {},
            "test": {}
        }

        # pour captum
        self.concat_layer = nn.Identity()
        
    def get_pooled_output(self, input_ids, attention_mask, pixel_values):
        """
        Return a pooled embedding from both modalities.
        For CLIP, use pooled_output; for BLIP, use CLS token of last_hidden_state.
        """
        self.embedder_model.to(self.device)
        
        if not self.multimodal_features:
            outputs = self.embedder_model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                pixel_values=pixel_values
            )
            image_features = outputs.image_embeds
            text_features = outputs.text_embeds
            combined_features = torch.cat((text_features, image_features), dim=1)
        else:
            combined_features = self.embedder_model.get_multimodal_features(
                input_ids=input_ids, 
                pixel_values=pixel_values,
                attention_mask=attention_mask
            )
            combined_features = combined_features / combined_features.norm(dim=-1, keepdim=True)    
    
        return combined_features
         
    def forward(self, input_ids, attention_mask, pixel_values):
        """
        Effectue une passe avant à travers le modèle pour obtenir les prédictions de labels.
        """
        self.embedder_model.eval()
        self.classifier.eval()

        with torch.no_grad():
            pooled_output = self.get_pooled_output(input_ids, attention_mask, pixel_values)
            pooled_output = self.concat_layer(pooled_output)
            logits = self.classifier(pooled_output)
            predictions = torch.argmax(logits, axis=1)
        
        return predictions

    def train_model(self, early_stopping=False, patience=5):
        # CHANGE 3: Added early_stopping and patience parameters (matching CustomClassifier signature)
        best_val_f1 = 0.
        epochs_without_improvement = 0
        best_model_state = None

        for epoch in range(self.num_epochs):
            self.embedder_model.train()
            self.classifier.train()

            # CHANGE 4: Added tqdm with lr display in description (matching CustomClassifier train loop style)
            train_loop = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch+1}/{self.num_epochs} [Train] "
                     f"(lr={[group['lr'] for group in self.optimizer.param_groups]})"
            )
            for batch in train_loop:

                label = batch['label'].to(self.device)
                images = batch['image'].to(self.device)
                texts = batch['text']
                
                # CHANGE 5: Separated text tokenization from image processing to avoid
                # double-normalizing images that are already preprocessed by the dataloader.
                # Previously: self.tokenizer(text=texts, images=images, ...) which re-processed images.
                # Now: tokenize text only, pass raw pixel_values separately (matching CustomClassifier).
                
                inputs = self.tokenizer(
                    text=texts,
                    return_tensors="pt", padding=True, truncation=True
                )
                for k, v in inputs.items():
                    inputs[k] = v.to(self.device)
                    
                self.optimizer.zero_grad()
                pooled_output = self.get_pooled_output(
                    inputs.input_ids, inputs.attention_mask, images  # images passed directly
                )

                pooled_output = self.concat_layer(pooled_output)
                logits = self.classifier(pooled_output)
                loss = self.loss_fn(logits, label)
                loss.backward()
                self.optimizer.step()

                train_loop.set_postfix(loss=loss.item())

                del pooled_output, logits
                torch.cuda.empty_cache()

            print(f"Epoch {epoch + 1}")
            train_accuracy, train_mean_macro_f1_score = self.evaluate_model(self.train_loader, "Train")
            self.performances["train"][f"epoch_{epoch+1}"] = {
                "accuracy": train_accuracy,
                "macro_f1": train_mean_macro_f1_score
            }

            if self.val_loader is not None:
                val_accuracy, val_f1 = self.evaluate_model(self.val_loader, "Val")
                self.performances["val"][f"epoch_{epoch+1}"] = {
                    "accuracy": val_accuracy,
                    "macro_f1": val_f1
                }

                # CHANGE 6: Early stopping on val_f1 instead of val_accuracy,
                # with copy.deepcopy of state (matching CustomClassifier)
                if early_stopping:
                    if val_f1 > best_val_f1:
                        best_val_f1 = val_f1
                        epochs_without_improvement = 0
                        best_model_state = copy.deepcopy(self.state_dict())
                    else:
                        epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        print(f"Early stopping triggered after {epoch+1} epochs")
                        break
                else:
                    best_model_state = copy.deepcopy(self.state_dict())
            else:
                best_model_state = copy.deepcopy(self.state_dict())

            # CHANGE 7: scheduler.step() moved inside epoch loop (was missing entirely before)
            self.scheduler.step()

        if best_model_state is not None:
            self.load_state_dict(best_model_state)
        self.save_model()

    def evaluate_model(self, loader, mode):
        self.embedder_model.eval()
        self.classifier.eval()
        accuracy = 0.
        predict_labels = np.array([])
        true_labels = np.array([])
        with torch.no_grad():
            for batch in tqdm(loader, desc=mode, unit="batch"):
                label = batch['label'].to(self.device)
                images = batch['image'].to(self.device)
                texts = batch['text']
                
                # CHANGE 5 (continued): same fix as in train_model — text-only tokenization
                inputs = self.tokenizer(
                    text=texts,
                    return_tensors="pt", padding=True, truncation=True
                )
                for k, v in inputs.items():
                    inputs[k] = v.to(self.device)
                    
                pooled_output = self.get_pooled_output(
                    inputs.input_ids, inputs.attention_mask, images
                )
                pooled_output = self.concat_layer(pooled_output)
                logits = self.classifier(pooled_output)
                predictions = torch.argmax(logits, axis=1)
                accuracy += torch.sum(predictions == label).item()
                predict_labels = np.append(predict_labels, predictions.cpu().numpy())
                true_labels = np.append(true_labels, label.cpu().numpy())

                del pooled_output, logits, predictions
                torch.cuda.empty_cache()
                
        accuracy /= len(loader.dataset)
        num_true_labels = len(np.unique(true_labels))
        macro_f1_scores = []
        for label in range(num_true_labels):
            label_pred = np.array(predict_labels) == label
            label_true = np.array(true_labels) == label
            macro_f1_scores.append(f1_score(label_true, label_pred, average='macro'))
        mean_macro_f1_score = np.mean(macro_f1_scores)
        print(f"{mode} Acc = {accuracy*100:.2f}%  {mode} Macro F1 = {mean_macro_f1_score*100:.2f}%")

        if mode == 'Test':
            self.performances["test"] = {
                "accuracy": accuracy,
                "macro_f1": mean_macro_f1_score
            }
            
        return accuracy, mean_macro_f1_score

    def _get_state_suffix(self):
        """Get suffix for sigmoid/relu variant."""
        if hasattr(self.config, 'sigmoid_or_relu_state') and self.config.sigmoid_or_relu_state != 'linearity':
            return f"_{self.config.sigmoid_or_relu_state}"
        return ""

    def _get_base_dir(self):
        """Get base directory for model checkpoints."""
        return f"{self.save_path}blue_checkpoints/{self.config.model_name}/BaselineModel"

    def save_model(self):
        base_dir = self._get_base_dir()
        os.makedirs(base_dir, exist_ok=True)
        suffix = self._get_state_suffix()
        
        torch.save(
            self.classifier.state_dict(), 
            f"{base_dir}/{self.embedder_model_name}_classifier_state_dict{suffix}.pth"
        )
        torch.save(
            self.embedder_model.state_dict(), 
            f"{base_dir}/{self.embedder_model_name}_state_dict{suffix}.pth"
        )
        print(f"Modèle sauvegardé dans {base_dir}")

    def load_model(self):
        base_dir = self._get_base_dir()
        suffix = self._get_state_suffix()
        
        classifier_path = f"{base_dir}/{self.embedder_model_name}_classifier_state_dict{suffix}.pth"
        embedder_path = f"{base_dir}/{self.embedder_model_name}_state_dict{suffix}.pth"
        
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
    
        perf_path = f"{base_dir}/{self.embedder_model_name}_performances{suffix}.json"
        if os.path.exists(perf_path):
            with open(perf_path, "r") as f:
                self.performances = json.load(f)
            print(f"Performances chargées depuis {perf_path}")
        else:
            self.performances = {}
            print("Aucune performance enregistrée trouvée.")
    
        gc.collect()
        torch.cuda.empty_cache()

    def save_performance_json(self):
        base_dir = self._get_base_dir()
        os.makedirs(base_dir, exist_ok=True)
        suffix = self._get_state_suffix()
        perf_path = f"{base_dir}/{self.embedder_model_name}_performances{suffix}.json"
        with open(perf_path, "w") as f:
            json.dump(self.performances, f, indent=4)
        print(f"Performances sauvegardées dans {perf_path}")