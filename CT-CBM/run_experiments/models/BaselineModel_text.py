import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
import gc
import os
import copy
import json
import torch.nn as nn

class BaselineModel(nn.Module):
    def __init__(self, embedder_model, classifier, train_loader, val_loader = None, test_loader = None, config = None, save_path = None, use_cls_token=False):
        super().__init__()  # Appel au constructeur de nn.Module

        self.embedder_model  = embedder_model
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
        self.optimizer = torch.optim.Adam(list(self.embedder_model.parameters()) + list(self.classifier.parameters()), lr=1e-5)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.num_epochs = self.config.num_epochs
        self.embedder_model_name = self.config.model_name
        self.best_acc_score = 0
        self.use_cls_token = use_cls_token
        self.embedder_model.to(self.device)
        self.classifier.to(self.device)
        
        # Dictionnaire pour stocker les performances
        self.performances = {
            "train": {},
            "val": {},
            "test": {}
        }
        
    def get_pooled_output(self, input_ids, attention_mask):
        """ get the embedding of the input text"""
        outputs = self.embedder_model(input_ids=input_ids, attention_mask=attention_mask)
        if self.use_cls_token:
            pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        else:
            pooled_output = outputs.last_hidden_state.mean(1)  # Mean pooling
        return pooled_output
        
    
    def forward(self, input_ids, attention_mask):
        """
        Effectue une passe avant à travers le modèle pour obtenir les prédictions de labels.

        Arguments:
        - input_ids (torch.Tensor): Les identifiants des entrées (séquences).
        - attention_mask (torch.Tensor): Le masque d'attention pour les séquences d'entrées.

        Retourne:
        - torch.Tensor: Les labels prédits pour les séquences d'entrée.
        """
        self.embedder_model.eval()  # Désactiver le dropout pour l'inférence
        self.classifier.eval()

        with torch.no_grad():
            # Passe à travers le modèle d'embedding
            outputs = self.embedder_model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Appliquer le pooling (moyenne des états cachés)
            if self.embedder_model_name == 'lstm':
                pooled_output = outputs.mean(1)
            elif self.use_cls_token:
                # use cls token
                pooled_output = outputs.last_hidden_state[:, 0, :]
            else:
                pooled_output = outputs.last_hidden_state.mean(1)
            
            # Passer les caractéristiques intégrées au classificateur
            logits = self.classifier(pooled_output)
            
            # Obtenir les prédictions finales
            predictions = torch.argmax(logits, axis=1)
        
        return predictions

    def train_model(self):
        for epoch in range(self.num_epochs):
            self.embedder_model.train()
            self.classifier.train()
            for batch in tqdm(self.train_loader, desc="Training", unit="batch"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                label = batch["label"].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.embedder_model(input_ids=input_ids, attention_mask=attention_mask)
                if self.use_cls_token:
                    pooled_output = outputs.last_hidden_state[:, 0, :]
                else:
                    pooled_output = outputs.last_hidden_state.mean(1)
                logits = self.classifier(pooled_output)
                loss = self.loss_fn(logits, label)
                loss.backward()
                self.optimizer.step()

                # Libérer la mémoire GPU après chaque batch
                del input_ids, attention_mask, label, outputs, pooled_output, logits
                torch.cuda.empty_cache()

            print(f"Epoch {epoch + 1}")
            # Évaluation sur le jeu de données d'entraînement
            train_accuracy, train_mean_macro_f1_score = self.evaluate_model(self.train_loader, "Train")
            self.performances["train"][f"epoch_{epoch+1}"] = {
                "accuracy": train_accuracy,
                "macro_f1": train_mean_macro_f1_score
            }

            
            
            # Évaluation sur le jeu de validation s'il est présent sinon on prend le modèle à la dernière itération
            if self.val_loader is not None:
                val_accuracy, mean_macro_f1_score = self.evaluate_model(self.val_loader, "Val")
                self.performances["val"][f"epoch_{epoch+1}"] = {
                    "accuracy": val_accuracy,
                    "macro_f1": mean_macro_f1_score
                }
                # we choose the best model on validation set
                if val_accuracy >= self.best_acc_score:
                    self.best_acc_score = val_accuracy
                    self.save_model()
        if self.val_loader is None:
            # Efficient car on ne save pas le modèle à chaque itération mais juste à la fin si le val loader n'existe pas
            self.save_model()

    def evaluate_model(self, loader, mode):
        self.embedder_model.eval()
        self.classifier.eval()
        accuracy = 0.
        predict_labels = np.array([])
        true_labels = np.array([])
        with torch.no_grad():
            for batch in tqdm(loader, desc=mode, unit="batch"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                label = batch["label"].to(self.device)
                outputs = self.embedder_model(input_ids=input_ids, attention_mask=attention_mask)
                if self.embedder_model_name == 'lstm':
                    pooled_output = outputs.mean(1)
                elif self.use_cls_token:
                    pooled_output = outputs.last_hidden_state[:, 0, :]
                else:
                    pooled_output = outputs.last_hidden_state.mean(1)
                logits = self.classifier(pooled_output)
                predictions = torch.argmax(logits, axis=1)
                accuracy += torch.sum(predictions == label).item()
                predict_labels = np.append(predict_labels, predictions.cpu().numpy())
                true_labels = np.append(true_labels, label.cpu().numpy())

                # Libérer la mémoire GPU après chaque batch
                del input_ids, attention_mask, label, outputs, pooled_output, logits, predictions
                torch.cuda.empty_cache()

        accuracy /= len(loader.dataset)
        num_true_labels = len(np.unique(true_labels))
        macro_f1_scores = []
        for label in range(num_true_labels):
            label_pred = np.array(predict_labels) == label
            label_true = np.array(true_labels) == label
            macro_f1_scores.append(f1_score(label_true, label_pred, average='macro'))
        mean_macro_f1_score = np.mean(macro_f1_scores)
        print(f"{mode} Acc = {accuracy*100} {mode} Macro F1 = {mean_macro_f1_score*100}")

        if mode=='Test':
            self.performances["test"] = {
                "accuracy": accuracy,
                "macro_f1": mean_macro_f1_score
            }
            
        return accuracy, mean_macro_f1_score

    def save_model(self):
        os.makedirs(f"{self.save_path}blue_checkpoints/{self.config.model_name}/BaselineModel", exist_ok=True)
        if self.config.sigmoid_or_relu_state != 'linearity':
            torch.save(self.classifier.state_dict(), f"{self.save_path}blue_checkpoints/{self.config.model_name}/BaselineModel/{self.embedder_model_name}_classifier_state_dict_{self.config.sigmoid_or_relu_state}.pth")
            torch.save(self.embedder_model.state_dict(), f"{self.save_path}blue_checkpoints/{self.config.model_name}/BaselineModel/{self.embedder_model_name}_state_dict_{self.config.sigmoid_or_relu_state}.pth")
        else:
            torch.save(self.classifier.state_dict(), f"{self.save_path}blue_checkpoints/{self.config.model_name}/BaselineModel/{self.embedder_model_name}_classifier_state_dict.pth")
            torch.save(self.embedder_model.state_dict(), f"{self.save_path}blue_checkpoints/{self.config.model_name}/BaselineModel/{self.embedder_model_name}_state_dict.pth")
            

    def load_model(self):
        if self.config.sigmoid_or_relu_state != 'linearity':
            self.classifier.load_state_dict(torch.load(f"{self.save_path}blue_checkpoints/{self.config.model_name}/BaselineModel/{self.embedder_model_name}_classifier_state_dict_{self.config.sigmoid_or_relu_state}.pth"))
            self.embedder_model.load_state_dict(torch.load(f"{self.save_path}blue_checkpoints/{self.config.model_name}/BaselineModel/{self.embedder_model_name}_state_dict_{self.config.sigmoid_or_relu_state}.pth"))
        
        else:
            self.classifier.load_state_dict(torch.load(f"{self.save_path}blue_checkpoints/{self.config.model_name}/BaselineModel/{self.embedder_model_name}_classifier_state_dict.pth"))
            self.embedder_model.load_state_dict(torch.load(f"{self.save_path}blue_checkpoints/{self.config.model_name}/BaselineModel/{self.embedder_model_name}_state_dict.pth"))
            
        self.embedder_model.to(self.device)
        self.classifier.to(self.device)
        self.embedder_model.eval()
        self.classifier.eval()
    
        # Charger les performances si le fichier existe
        performance_file = os.path.join(self.save_path, "blue_checkpoints", self.config.model_name, "BaselineModel", f"{self.embedder_model_name}_performances_{self.config.sigmoid_or_relu_state}.json")
        if os.path.exists(performance_file):
            with open(performance_file, "r") as f:
                self.performances = json.load(f)
            print(f"Performances chargées depuis {performance_file}")
        else:
            self.performances = {}
            print("Aucune performance enregistrée trouvée.")
    
        # Appeler le ramasse-miettes
        gc.collect()
        torch.cuda.empty_cache()


    def save_performance_json(self):
        """Sauvegarder le dictionnaire des performances dans un fichier JSON."""
        # Chemin complet pour sauvegarder le fichier JSON
        performance_file = os.path.join(self.save_path, "blue_checkpoints", self.config.model_name, "BaselineModel", f"{self.embedder_model_name}_performances_{self.config.sigmoid_or_relu_state}.json")
        with open(performance_file, "w") as f:
            json.dump(self.performances, f, indent=4)
        print(f"Performances sauvegardées dans {performance_file}")


    
