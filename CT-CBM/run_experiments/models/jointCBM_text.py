import os
import time
import torch
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from tabulate import tabulate

from models.utils import ElasticNetLinearLayer


class JointModel:
    def __init__(self, model, tokenizer, ModelXtoCtoY_layer, config, train_loader, val_loader, num_epochs=2, use_cls_token = True, num_discoverded_concepts_ = 1, linear_layer = None):
        self.embedder_model = model
        self.embedder_tokenizer = tokenizer
        self.ModelXtoCtoY_layer = ModelXtoCtoY_layer
        self.config = config
        self.device = config.device
        self.model_name = config.model_name
        self.num_epochs = config.num_epochs
        self.lambda_XtoC = config.lambda_XtoC
        # self.train_loader = train_loader
        self.val_loader = val_loader
        self.use_cls_token = use_cls_token
        self.linear_layer = linear_layer

        self.optimizer = Adam(
            list(self.embedder_model.parameters()) + list(self.ModelXtoCtoY_layer.parameters()),
            lr=1e-5)

        self.loss_concept_function = BCEWithLogitsLoss()
        self.loss_fn = CrossEntropyLoss()
        self.embedder_model.to(self.device)
        self.ModelXtoCtoY_layer.to(self.device)
        
        #self.classifier = None
        
        # for pipeline purpose
        self.concepts_name = [] 
        self.alternate_save = False
        self.strategy = None # 'random', 'tcavs', 'lig', 'frequences'
        
    # OK
    def get_pooled_output(self, input_ids, attention_mask):
        """ get the embedding of the input text"""
        outputs = self.embedder_model(input_ids=input_ids, attention_mask=attention_mask)
        if self.use_cls_token:
            pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        else:
            pooled_output = outputs.last_hidden_state.mean(1)  # Mean pooling
        return pooled_output

    # OK
    def get_XtoY_output(self, pooled_output):
        """
        just a useful function to get the output of the XtoY layer
        """
        outputs = self.ModelXtoCtoY_layer(pooled_output)
        XtoC_output = outputs[1:] 
        XtoY_output = outputs[0:1]
        return XtoY_output, XtoC_output
      
    def forward(self, batch):
        """
        idea : function for a batch, forward for model joint without residual layer
        """
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["label"].to(self.device)

        outputs = self.embedder_model(input_ids=input_ids, attention_mask=attention_mask)
        if self.use_cls_token:
            pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        else:
            pooled_output = outputs.last_hidden_state.mean(1)

        outputs = self.ModelXtoCtoY_layer(pooled_output)
        return outputs, labels

    # NEW
    def forward_linear(self, input_ids, attention_mask):
        pooled_output = self.get_pooled_output(input_ids, attention_mask)
        linear_output = self.linear_layer(pooled_output)
        XtoY_output, XtoC_output = self.get_XtoY_output(pooled_output)
        final_output = linear_output + XtoY_output[0]
        return final_output, XtoC_output, pooled_output

    # used in LIG 
    def forward_residual_layer(self, input_ids, attention_mask):
        """Necessary forward for attribution calculation purpose for the residual layer retropropagation"""
        pooled_output = self.get_pooled_output(input_ids, attention_mask)
        linear_output = self.linear_layer(pooled_output)
        return linear_output

    # used in LIG on concept
    def forward_per_concept(self, input_ids, attention_mask, concept_number = 1):
        """Necessary forward for attribution calculation purpose for the token importance per concepts """
        pooled_output = self.get_pooled_output(input_ids, attention_mask)
        # concept_output = self.ModelXtoCtoY_layer.first_model.forward_fc_layer(pooled_output)
        concept_output = self.ModelXtoCtoY_layer.first_model.all_fc[concept_number](pooled_output)
        # concept_output est le logit (Z) et non la valeur 0 ou 1
        # print("concept_output", concept_output)
        # print("concept_output.shape", concept_output.shape)
        return concept_output

    #OK
    def reset_classifier(self):
        """Réinitialise les poids de la couche classifier (ElasticNetLinearLayer)."""
        if self.classifier is not None:
            self.classifier.reset_parameters()

    def train_model(self, train_loader, val_loader):
        start_time = time.time()
        best_acc_score = 0
        kept_epoch = 0

        # Dictionnaire pour stocker les prédictions et labels par concept
        concept_pred_labels = {concept: [] for concept in self.concepts_name}
        concept_true_labels = {concept: [] for concept in self.concepts_name}

        # Listes pour l'accuracy totale sur tous les concepts
        all_concepts_pred = []
        all_concepts_true = []
        
        for epoch in range(self.num_epochs):
            self.embedder_model.train()
            self.ModelXtoCtoY_layer.train()
            
            for batch in tqdm(train_loader, desc="Training", unit="batch"):
                outputs, labels = self.forward(batch)
                XtoC_output = outputs[1:]
                XtoY_output = outputs[0:1]

                # # XtoC loss
                selected_columns = self.concepts_name
                # print("selected_columns",selected_columns)
                # print("labels", labels)
                selected_tensors = torch.stack([batch[col] for col in selected_columns], dim=1)
                # print(selected_tensors)
                concept_labels = selected_tensors
                concept_labels = torch.t(concept_labels)
                concept_labels = concept_labels.contiguous().view(-1)
                # TODO after : Conditional logic for selecting loss function
                # if concept_labels.max() > 1:  # Multi-class concept classification (need to have the max on all label and not only the label of one batch so use config or something else tahn below way)
                #     XtoC_loss = CrossEntropyLoss()(XtoC_logits, concept_labels.to(self.device))
                # else:  # Binary concept classification

                XtoC_logits = torch.cat(XtoC_output, dim=0).squeeze(dim=1)  # Remove the extra dimension
                concept_labels = concept_labels.to(self.device).float() # Ensure concept_labels are of type float

                #sigmoid already intregrated in loss function (good for numerical stability to do this way)
                XtoC_loss = self.loss_concept_function(XtoC_logits, concept_labels.to(self.device)) 
                
                # XtoY loss
                # print("XtoY_output [0]", XtoY_output)
                # print("XtoC_output", XtoC_output)


                # identify linear layer in ModelXtoCtoY_layer then get the weight
                # use the parameters to do the regularisation part of elastic net
                l1_norm = torch.norm(self.ModelXtoCtoY_layer.sec_model.linear.weight, p=1)
                l2_norm = torch.norm(self.ModelXtoCtoY_layer.sec_model.linear.weight, p=2)

                XtoY_loss = self.loss_fn(XtoY_output[0], labels) + self.config.alpha * (self.config.l1_ratio * l1_norm + (1 - self.config.l1_ratio) * l2_norm)
                loss = XtoC_loss * self.lambda_XtoC + XtoY_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Accuracy et F1 par concept
                for i, concept in enumerate(self.concepts_name):
                    # print("torch.sigmoid(XtoC_output[i])",torch.sigmoid(XtoC_output[i]))                    
                    concept_pred = torch.round(torch.sigmoid(XtoC_output[i])).cpu().detach().numpy()
                    # print(concept_pred)
                    # print(torch.argmax(torch.softmax(XtoC_output[i], dim=1), axis=1).cpu().numpy()) # to pass in multi-class concept classification
                    concept_true = batch[concept].cpu().numpy()
                    
                    concept_pred_labels[concept].extend(concept_pred)
                    concept_true_labels[concept].extend(concept_true)

                    # Ajout aux listes globales pour l'accuracy totale
                    all_concepts_pred.extend(concept_pred)
                    all_concepts_true.extend(concept_true)
                    
            # Calcul de l'accuracy et du F1 par concept
            concept_accuracies = {}
            concept_f1_scores = {}
            for concept in self.concepts_name:
                concept_accuracies[concept] = accuracy_score(concept_true_labels[concept], concept_pred_labels[concept])
                concept_f1_scores[concept] = f1_score(concept_true_labels[concept], concept_pred_labels[concept], average='macro')

            # Calcul de l'accuracy totale sur tous les concepts
            total_concept_accuracy = accuracy_score(all_concepts_true, all_concepts_pred)
                
            # Evaluate on training data
            train_metrics = self.evaluate_model(train_loader)
            print(f"Epoch {epoch + 1}: Train Acc = {train_metrics['task_global_accuracy']*100} Train Macro F1 = {train_metrics['task_global_macro_f1_score']*100}")
    
            # Evaluate on validation data
            val_metrics = self.evaluate_model(val_loader)
            print(f"Epoch {epoch + 1}: Val Acc = {val_metrics['task_global_accuracy']*100} Val Macro F1 = {val_metrics['task_global_macro_f1_score']*100}")
            
            if val_metrics['task_global_accuracy'] > best_acc_score:
                kept_epoch = epoch
                best_acc_score = val_metrics['task_global_accuracy']
                best_F1_score = val_metrics['task_global_macro_f1_score']
                # Affichage de l'accuracy totale
                best_total_concept_accuracy = total_concept_accuracy
                
                total_concept_accuracy_kept = total_concept_accuracy # take the one on the best model according to val acc
                self.save_model()

        total_duration = time.time() - start_time
        print(f"Total training duration: {total_duration:.2f} seconds")
        print(f"Best accuracy found at epoch {kept_epoch + 1}")
        print(f"Best accuracy: {best_acc_score*100:.2f}%")
        print(f"Epoch {kept_epoch + 1}, F1 associated: {best_F1_score*100:.2f}%")
        print(f"Epoch {kept_epoch + 1} Accuracy et F1 par concept :")
        for concept in self.concepts_name:
            print(f"{concept}: Accuracy = {concept_accuracies[concept]*100:.2f}%, F1 = {concept_f1_scores[concept]*100:.2f}%")
        print(f"Epoch {kept_epoch + 1}: Total Concept Accuracy = {best_total_concept_accuracy * 100:.2f}%")


        return total_concept_accuracy_kept, concept_accuracies, concept_f1_scores
    
    def evaluate_model(self, data_loader, verbose=False, metrics_on_concepts = False):
        """ This v2 version computes task accuracy and F1-score for the main task, 
        similar to CBE-PLMS paper, while also allowing concept classification evaluation."""
        self.embedder_model.eval()
        self.ModelXtoCtoY_layer.eval()
        
        # Variables pour les métriques sur la tâche principale
        total_accuracy = 0
        predict_labels = np.array([])
        true_labels = np.array([])
        
        # Dictionnaires pour les métriques sur les concepts
        concept_pred_labels = {concept: [] for concept in self.concepts_name}
        concept_true_labels = {concept: [] for concept in self.concepts_name}
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating", unit="batch"):
                outputs, labels = self.forward(batch)
                XtoC_output = outputs[1:]
                XtoY_output = outputs[0:1]

                # Classification accuracy
                predictions = torch.argmax(XtoY_output[0], axis=1)
                total_accuracy += torch.sum(predictions == labels).item()
                predict_labels = np.append(predict_labels, predictions.cpu().numpy())
                true_labels = np.append(true_labels, labels.cpu().numpy())

                # Si le calcul des métriques sur les concepts est activé
                if metrics_on_concepts:
                    for i, concept in enumerate(self.concepts_name):
                        concept_pred = torch.round(torch.sigmoid(XtoC_output[i])).cpu().detach().numpy()
                        concept_true = batch[concept].cpu().numpy()
                        concept_pred_labels[concept].extend(concept_pred)
                        concept_true_labels[concept].extend(concept_true)
                        
        # même methode de calcul que CBE-PLMs pour acc et F1               
        total_accuracy /= len(data_loader.dataset)

        # Calcul du F1 global pour la tâche principale (similaire à v1)
        num_labels = len(np.unique(true_labels))
        macro_f1_scores = [
            f1_score(true_labels == label, predict_labels == label, average="macro") for label in range(num_labels)
        ]
        mean_macro_f1_score = np.mean(macro_f1_scores)

        # Calcul de l'accuracy par classe
        num_classes = len(np.unique(true_labels))
        accuracies_per_class = {}
        for i in range(num_classes):
            class_pred = (predict_labels == i)
            class_true = (true_labels == i)
            if np.sum(class_true) > 0:  # Pour éviter la division par zéro
                class_accuracy = np.sum(class_pred & class_true) / np.sum(class_true)
            else:
                class_accuracy = 0
            accuracies_per_class[i] = class_accuracy

        # Calcul du F1-score par classe
        f1_scores_per_class = []
        for i in range(num_classes):
            f1 = f1_score(true_labels == i, predict_labels == i, average="macro")
            f1_scores_per_class.append(f1)

        # Résultats globaux pour la tâche principale
        results = {
            "task_global_accuracy": total_accuracy,
            "task_global_macro_f1_score": mean_macro_f1_score,
            "accuracies_per_class": accuracies_per_class,
            "f1_scores_per_class": f1_scores_per_class,
        }

        # Calcul des métriques par concept si activé
        if self.config.eval_concepts:
            concept_accuracies = {}
            concept_f1_scores = {}
            for concept in self.concepts_name:
                concept_accuracies[concept] = accuracy_score(concept_true_labels[concept], concept_pred_labels[concept])
                concept_f1_scores[concept] = f1_score(concept_true_labels[concept], concept_pred_labels[concept], average="macro")
            
            # Calcul des moyennes des métriques des concepts
            mean_concept_accuracy = np.mean(list(concept_accuracies.values()))
            mean_concept_f1_score = np.mean(list(concept_f1_scores.values()))
            
            results.update({
                "concept_accuracies": concept_accuracies,
                "concept_f1_scores": concept_f1_scores,
                "mean_concept_accuracy": mean_concept_accuracy,
                "mean_concept_f1_score": mean_concept_f1_score
            })

        if verbose:
            print(f"Test Acc = {results['task_global_accuracy'] * 100:.2f}, Test Macro F1 = {results['task_global_macro_f1_score'] * 100:.2f}")
            print(f"Accuracy par classe: {results['accuracies_per_class']}")
            print(f"F1-score par classe: {results['f1_scores_per_class']}")
            if metrics_on_concepts:
                print(f"Moyenne des métriques des concepts : Accuracy = {results['mean_concept_accuracy'] * 100:.2f}%, F1 = {results['mean_concept_f1_score'] * 100:.2f}%")
                print("Métriques par concept :")
                for concept in self.concepts_name:
                    print(f"{concept}: Accuracy = {concept_accuracies[concept] * 100:.2f}%, F1 = {concept_f1_scores[concept] * 100:.2f}%")
        
        return results         

    def evaluate_model_v5(self, data_loader, verbose=False, metrics_on_concepts=False):
        """ This v5 version computes task accuracy and F1-score for the main task, 
        similar to CBE-PLMS paper, while also allowing concept classification evaluation."""
        self.embedder_model.eval()
        self.ModelXtoCtoY_layer.eval()
        
        # Variables pour les métriques sur la tâche principale
        total_accuracy = 0
        predict_labels = np.array([])
        true_labels = np.array([])
        
        # Dictionnaires pour les métriques sur les concepts
        concept_pred_labels = {concept: [] for concept in self.concepts_name}
        concept_true_labels = {concept: [] for concept in self.concepts_name}
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating", unit="batch"):
                outputs, labels = self.forward(batch)
                XtoC_output = outputs[1:]
                XtoY_output = outputs[0:1]
    
                # Classification accuracy sur la tâche principale
                predictions = torch.argmax(XtoY_output[0], axis=1)
                total_accuracy += torch.sum(predictions == labels).item()
                predict_labels = np.append(predict_labels, predictions.cpu().numpy())
                true_labels = np.append(true_labels, labels.cpu().numpy())
    
                # Si le calcul des métriques sur les concepts est activé
                if metrics_on_concepts:
                    for i, concept in enumerate(self.concepts_name):
                        # Vérifier que le concept existe dans le batch
                        if concept in batch:
                            concept_pred = torch.round(torch.sigmoid(XtoC_output[i])).cpu().detach().numpy()
                            concept_true = batch[concept].cpu().numpy()
                            concept_pred_labels[concept].extend(concept_pred)
                            concept_true_labels[concept].extend(concept_true)
                        else:
                            print(f"Le concept '{concept}' n'est pas présent dans le batch courant, il sera ignoré pour l'évaluation.")
                            
        # Calcul des métriques pour la tâche principale
        total_accuracy /= len(data_loader.dataset)
        num_labels = len(np.unique(true_labels))
        macro_f1_scores = [
            f1_score(true_labels == label, predict_labels == label, average="macro") for label in range(num_labels)
        ]
        mean_macro_f1_score = np.mean(macro_f1_scores)
    
        # Calcul de l'accuracy et du F1 par classe pour la tâche principale
        num_classes = len(np.unique(true_labels))
        accuracies_per_class = {}
        for i in range(num_classes):
            class_pred = (predict_labels == i)
            class_true = (true_labels == i)
            if np.sum(class_true) > 0:
                class_accuracy = np.sum(class_pred & class_true) / np.sum(class_true)
            else:
                class_accuracy = 0
            accuracies_per_class[i] = class_accuracy
    
        f1_scores_per_class = []
        for i in range(num_classes):
            f1 = f1_score(true_labels == i, predict_labels == i, average="macro")
            f1_scores_per_class.append(f1)
    
        results = {
            "task_global_accuracy": total_accuracy,
            "task_global_macro_f1_score": mean_macro_f1_score,
            "accuracies_per_class": accuracies_per_class,
            "f1_scores_per_class": f1_scores_per_class,
        }
    
        # Calcul des métriques par concept si activé
        if self.config.eval_concepts:
            concept_accuracies = {}
            concept_f1_scores = {}
            # Pour chaque concept, on vérifie si des exemples ont été collectés
            for concept in self.concepts_name:
                if len(concept_true_labels[concept]) > 0:
                    concept_accuracies[concept] = accuracy_score(concept_true_labels[concept], concept_pred_labels[concept])
                    concept_f1_scores[concept] = f1_score(concept_true_labels[concept], concept_pred_labels[concept], average="macro")
                else:
                    print(f"Le concept '{concept}' n'est pas présent dans le dataloader, ses métriques ne sont pas calculées.")
            
            # Calcul des moyennes en considérant uniquement les concepts évalués
            mean_concept_accuracy = np.mean(list(concept_accuracies.values())) if concept_accuracies else 0
            mean_concept_f1_score = np.mean(list(concept_f1_scores.values())) if concept_f1_scores else 0
            
            results.update({
                "concept_accuracies": concept_accuracies,
                "concept_f1_scores": concept_f1_scores,
                "mean_concept_accuracy": mean_concept_accuracy,
                "mean_concept_f1_score": mean_concept_f1_score
            })
    
        if verbose:
            print(f"Test Acc = {results['task_global_accuracy'] * 100:.2f}, Test Macro F1 = {results['task_global_macro_f1_score'] * 100:.2f}")
            print(f"Accuracy par classe: {results['accuracies_per_class']}")
            print(f"F1-score par classe: {results['f1_scores_per_class']}")
            if metrics_on_concepts:
                print(f"Moyenne des métriques des concepts : Accuracy = {results['mean_concept_accuracy'] * 100:.2f}%, F1 = {results['mean_concept_f1_score'] * 100:.2f}%")
                print("Métriques par concept :")
                for concept in self.concepts_name:
                    if concept in concept_accuracies:
                        print(f"{concept}: Accuracy = {concept_accuracies[concept] * 100:.2f}%, F1 = {concept_f1_scores[concept] * 100:.2f}%")
        
        return results


    
    def run(self, train_loader, val_loader, test_loader, verbose=False, meta_on_concept=False):
        """
        Exécute l'entraînement et l'évaluation du modèle sur les ensembles d'entraînement, validation et test.
        """
        # Entraînement du modèle
        task_concepts_perf = self.train_model(train_loader, val_loader)

        # Évaluation sur les ensembles d'entraînement, de validation et de test
        train_metrics = self.evaluate_model(train_loader, "train")
        val_metrics = self.evaluate_model(val_loader, "val", metrics_on_concepts=False)
        test_metrics = self.evaluate_model(test_loader, "test", metrics_on_concepts = False)

        # Compilation des résultats
        run_info = {
            "train_acc": train_metrics["task_global_accuracy"],
            "val_acc": val_metrics["task_global_accuracy"],
            "test_acc": test_metrics["task_global_accuracy"],
            "f1_train": train_metrics["task_global_macro_f1_score"],
            "f1_val": val_metrics["task_global_macro_f1_score"],
            "f1_test": test_metrics["task_global_macro_f1_score"],
            "cls_acc_train": train_metrics["accuracies_per_class"],
            "cls_acc_val": val_metrics["accuracies_per_class"],
            "cls_acc_test": test_metrics["accuracies_per_class"],
            "cls_f1_train": train_metrics["f1_scores_per_class"],
            "cls_f1_val": val_metrics["f1_scores_per_class"],
            "cls_f1_test": test_metrics["f1_scores_per_class"]
        }

        # Affichage des résultats si verbose est activé
        if verbose:
            # Préparation des données pour tabulate
            table = [
                ['Metric', 'Train', 'Validation', 'Test'],
                ['Global Accuracy', f"{run_info['train_acc']:.2f}", f"{run_info['val_acc']:.2f}", f"{run_info['test_acc']:.2f}"],
                ['Global F1 Score', f"{run_info['f1_train']:.2f}", f"{run_info['f1_val']:.2f}", f"{run_info['f1_test']:.2f}"],
                ['Class Accuracies', str(run_info['cls_acc_train']), str(run_info['cls_acc_val']), str(run_info['cls_acc_test'])],
                ['Class F1 Scores', str(run_info['cls_f1_train']), str(run_info['cls_f1_val']), str(run_info['cls_f1_test'])]
            ]

            # Affichage des résultats avec tabulate
            print(tabulate(table, headers='firstrow', tablefmt='grid'))

        if meta_on_concept:
            return run_info, task_concepts_perf

        return run_info

    # ---------------------------------- ADDED METHOD FOR RESIDUAL
    # NEW
    def reset_residual_layer(self):
        """Reset the residual layer (linear_layer).
        TODO : EVALUTE the need to do this
        """
        if isinstance(self.linear_layer, torch.nn.Linear):
            self.linear_layer.reset_parameters()
        else:
            # TODO: For other types of layers (ReLU, etc.), adjust initialization accordingly
            for layer in self.linear_layer.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

    # NEW
    def train_residual_layer(self, train_loader, num_epochs):
        """Train the model using the CAVs."""
        print("-------- Training the residual layer ----------")

        # Reset the residual layer (linear_layer) before each iteration
        self.reset_residual_layer()
        
        self.embedder_model.eval()
        self.ModelXtoCtoY_layer.eval() 

        self.residual_optimizer = torch.optim.Adam(self.linear_layer.parameters(), lr=self.config.lr)

        for epoch in range(num_epochs):
            self.linear_layer.train()  

            train_accuracy = 0
            predict_labels = np.array([])
            true_labels = np.array([])

            for batch in tqdm(train_loader, desc="Train", unit="batch"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                label = batch["label"].to(self.device)

                final_output, XtoC_output, pooled_output = self.forward_linear(input_ids, attention_mask)

                loss = self.linear_layer.ridge_loss(final_output, label)
                self.residual_optimizer.zero_grad()
                loss.backward()
                self.residual_optimizer.step()

                predictions = torch.argmax(final_output, axis=1)
                train_accuracy += torch.sum(predictions == label).item()
                predict_labels = np.append(predict_labels, predictions.cpu().numpy())
                true_labels = np.append(true_labels, label.cpu().numpy())

                # Free GPU memory after each iteration
                del final_output, XtoC_output, pooled_output
                torch.cuda.empty_cache()

            train_accuracy /= len(train_loader.dataset)

            num_labels = len(np.unique(true_labels))

            macro_f1_scores = []
            for label in range(num_labels):
                label_pred = np.array(predict_labels) == label
                label_true = np.array(true_labels) == label
                macro_f1_scores.append(f1_score(label_true, label_pred, average='macro'))
            mean_macro_f1_score = np.mean(macro_f1_scores)
            macro_f1 = mean_macro_f1_score

            print(f"Epoch {epoch + 1}: Train Acc = {train_accuracy * 100}, Macro F1 = {macro_f1 * 100}")

            if self.alternate_save == True:
                os.makedirs(f"{self.config.SAVE_PATH}blue_checkpoints/{self.model_name}/Our_CBM_joint", exist_ok=True)
                torch.save(self.linear_layer.state_dict(), f"{self.config.SAVE_PATH}blue_checkpoints/{self.config.model_name}/Our_CBM_joint/{self.model_name}_residual_layer_joint_strategy_{self.strategy}_{self.iteration}.pth")
            else:
                torch.save(self.linear_layer.state_dict(), f"{self.config.SAVE_PATH}blue_checkpoints/{self.config.model_name}/jointCBM/{self.model_name}_residual_layer_joint_strategy.pth")                

    # NEW
    def evaluate_model_with_residual(self, data_loader, dataset_name =".."):
        """Evaluate the model."""
        print("-------- Evaluating the residual layer ----------")

        self.embedder_model.eval()
        self.ModelXtoCtoY_layer.eval()
        self.linear_layer.eval()

        criterion = CrossEntropyLoss()

        val_loss = 0
        val_accuracy = 0
        predict_labels = np.array([])
        true_labels = np.array([])

        total_accuracy = 0
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc= dataset_name, unit="batch"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                label = batch["label"].to(self.device)

                final_output, XtoC_output, _ = self.forward_linear(input_ids, attention_mask)
                loss = criterion(final_output, label)
                val_loss += loss.item()

                predictions = torch.argmax(final_output, axis=1)
                val_accuracy += torch.sum(predictions == label).item()
                predict_labels = np.append(predict_labels, predictions.cpu().numpy())
                true_labels = np.append(true_labels, label.cpu().numpy())

            val_loss /= len(data_loader)
            val_accuracy /= len(data_loader.dataset)

            # Macro F1 score
            num_labels = len(np.unique(true_labels))
            macro_f1_scores = []
            for label in range(num_labels):
                label_pred = np.array(predict_labels) == label
                label_true = np.array(true_labels) == label
                macro_f1_scores.append(f1_score(label_true, label_pred, average='macro'))
            mean_macro_f1_score = np.mean(macro_f1_scores)
            val_macro_f1 = mean_macro_f1_score
            

        # # # TODO: add the compute of des accuracy par classe
        # # Calcul de l'accuracy par classe
        # num_classes = len(np.unique(true_labels))
        # accuracies_per_class = {}
        # for i in range(num_classes):
        #     class_pred = (predict_labels == i)
        #     class_true = (true_labels == i)
        #     if np.sum(class_true) > 0:  # Pour éviter la division par zéro
        #         class_accuracy = np.sum(class_pred & class_true) / np.sum(class_true)
        #     else:
        #         class_accuracy = 0
        #     accuracies_per_class[i] = class_accuracy

        # # Calcul du F1-score par classe
        # f1_scores_per_class = []
        # for i in range(num_classes):
        #     f1 = f1_score(true_labels == i, predict_labels == i, average="macro")
        #     f1_scores_per_class.append(f1)
        
        print(f"{dataset_name} Loss: {val_loss}, {dataset_name} Acc: {val_accuracy * 100}, Macro F1: {val_macro_f1 * 100}")
            
        return val_loss, val_accuracy, val_macro_f1
    
    def save_model(self):
        if self.alternate_save:
            os.makedirs(f"{self.config.SAVE_PATH}blue_checkpoints/{self.model_name}/Our_CBM_joint", exist_ok=True)
            torch.save(self.embedder_model.state_dict(), f"{self.config.SAVE_PATH}blue_checkpoints/{self.model_name}/Our_CBM_joint/{self.model_name}_embedder_state_dict_{self.strategy}_{self.iteration}.pth")
            torch.save(self.ModelXtoCtoY_layer.state_dict(), f"{self.config.SAVE_PATH}blue_checkpoints/{self.model_name}/Our_CBM_joint/{self.model_name}_ModelXtoCtoY_layer_state_dict_{self.strategy}_{self.iteration}.pth")
        else:
            os.makedirs(f"{self.config.SAVE_PATH}blue_checkpoints/{self.model_name}/jointCBM", exist_ok=True)
            torch.save(self.embedder_model.state_dict(), f"{self.config.SAVE_PATH}blue_checkpoints/{self.model_name}/jointCBM/{self.model_name}_state_dict.pth")
            torch.save(self.ModelXtoCtoY_layer.state_dict(), f"{self.config.SAVE_PATH}blue_checkpoints/{self.model_name}/jointCBM/{self.model_name}_ModelXtoCtoY_layer_state_dict.pth")

    def load_model(self):
        if self.alternate_save:
            self.embedder_model.load_state_dict(torch.load(f"{self.config.SAVE_PATH}blue_checkpoints/{self.model_name}/Our_CBM_joint/{self.model_name}_embedder_state_dict_{self.strategy}_{self.iteration}.pth"))
            self.ModelXtoCtoY_layer.load_state_dict(torch.load(f"{self.config.SAVE_PATH}blue_checkpoints/{self.model_name}/Our_CBM_joint/{self.model_name}_ModelXtoCtoY_layer_state_dict_{self.strategy}_{self.iteration}.pth"))
            self.embedder_model.to(self.device)
            self.ModelXtoCtoY_layer.to(self.device)
            
            print('ModelXtoCtoY_layer loadef fom :', f"{self.config.SAVE_PATH}blue_checkpoints/{self.model_name}/Our_CBM_joint/{self.model_name}_ModelXtoCtoY_layer_state_dict_{self.strategy}_{self.iteration}.pth")
            print('embedder_model loadef fom :', f"{self.config.SAVE_PATH}blue_checkpoints/{self.model_name}/Our_CBM_joint/{self.model_name}_embedder_state_dict_{self.strategy}_{self.iteration}.pth")

            
            if self.linear_layer is not None:
                self.linear_layer.load_state_dict(torch.load(f"{self.config.SAVE_PATH}blue_checkpoints/{self.config.model_name}/Our_CBM_joint/{self.model_name}_residual_layer_joint_strategy_{self.strategy}_{self.iteration}.pth"))
                self.linear_layer.to(self.device)
        else:
            self.embedder_model.load_state_dict(torch.load(f"{self.config.SAVE_PATH}blue_checkpoints/{self.model_name}/jointCBM/{self.model_name}_state_dict.pth"))
            self.ModelXtoCtoY_layer.load_state_dict(torch.load(f"{self.config.SAVE_PATH}blue_checkpoints/{self.model_name}/jointCBM/{self.model_name}_ModelXtoCtoY_layer_state_dict.pth"))
            self.embedder_model.to(self.device)
            self.ModelXtoCtoY_layer.to(self.device)
            if self.linear_layer is not None:
                self.linear_layer.load_state_dict(torch.load(f"{self.config.SAVE_PATH}blue_checkpoints/{self.config.model_name}/jointCBM/{self.model_name}_residual_layer_joint_strategy.pth"))
                self.linear_layer.to(self.device)   
