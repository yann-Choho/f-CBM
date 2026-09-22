import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
import torch.nn.functional as F
from torchvision import transforms, datasets
from tqdm import tqdm
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold
import numpy as np
import glob
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score, classification_report
from transformers import BlipProcessor, BlipModel, AutoProcessor
from transformers import CLIPProcessor, CLIPModel, AutoProcessor
import re
from functools import lru_cache
import multiprocessing
import mmap
import json
import copy

import matplotlib.pyplot as plt

class CustomClassifier(nn.Module):
    def __init__(self, device, class_dict, base_model, processor, num_classes, classifier_type='simple', multimodal_features=False, frozen_backbone=False, modality='multi'):
        super(CustomClassifier, self).__init__()
        self.base_model = base_model
        self.processor = processor
        self.class_dict = class_dict
        self.device = device
        self.num_classes = num_classes

        self.frozen_backbone = frozen_backbone
        self.multimodal_features = multimodal_features
        self.combine_type = modality

        if(classifier_type == 'simple'):

            if(self.multimodal_features or self.combine_type=='text' or self.combine_type=='image'):
                # if multimodal features are used (so one vector for both modalities, only available with BLIP)
                # or if only text or image concepts are used, then just one vector as output from CLIP/BLIP
                #if(self.combine_type=='text'):
                #    self.classifier = nn.Linear(base_model.text_model.config.hidden_size, self.num_classes)
                #elif(self.combine_type=='image'):
                #    self.classifier = nn.Linear(base_model.vision_model.config.hidden_size, self.num_classes)
                #else:
                self.classifier = nn.Linear(base_model.config.projection_dim, self.num_classes)
            else:
                # text and image embeddings are produced separately, and are combined into one concept
                self.classifier = nn.Linear(base_model.config.projection_dim*2, self.num_classes)
                # self.classifier = nn.Sequential(
                #                         nn.Dropout(p=0.3),                  # Dropout applied first to input features
                #                         nn.Linear(base_model.config.projection_dim*2, self.num_classes)  # Linear classification layer
                #                         )

        else:
            self.classifier = nn.Sequential(
                nn.Linear(base_model.config.projection_dim * 2, 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, num_classes)
            )
        

        # Freeze the BLIP model
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

        
    def forward(self, input_ids, attention_mask, pixel_values):
        if(not self.multimodal_features):
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
            if(self.combine_type=='text'):
                combined_features = outputs.text_embeds
                #combined_features = output['text_model_output']["pooler_output"]
            elif(self.combine_type=='image'):
                combined_features = outputs.image_embeds
                #combined_features = outputs['vision_model_output']["pooler_output"]
            else:
                image_features = outputs.image_embeds
                text_features = outputs.text_embeds
                combined_features = torch.cat((text_features, image_features), dim=1)
        else:
            combined_features = self.base_model.get_multimodal_features(input_ids=input_ids, pixel_values=pixel_values,attention_mask=attention_mask)
            combined_features = combined_features / combined_features.norm(dim=-1, keepdim=True)

        logits = self.classifier(combined_features)

        return logits # return raw logits for crossentropyloss

    def predict(self, input_ids, attention_mask, pixel_values):
        """
        Make predictions using the model.
        
        Args:
        - input_ids: Tensor of input IDs
        - attention_mask: Tensor of attention mask
        - pixel_values: Tensor of pixel values

        Returns:
        - predicted_classes: Tensor of predicted class indices
        - probabilities: Tensor of class probabilities
        """
        self.to(self.device)
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient computation
            # Get logits from the forward pass
            logits = self(input_ids, attention_mask, pixel_values)
            
            # Apply softmax to get probabilities
            probabilities = F.softmax(logits, dim=1)
            
            # Get the predicted class (index of the highest probability)
            predicted_classes = torch.argmax(probabilities, dim=1)
        
        return [label for label in predicted_classes], probabilities
    
    def predict_contrastive(self, images, class_names):
        """
        Make predictions using the contrastively trained model.
        
        Args:
        - images: Tensor of images
        - class_names: List of class names (strings)
        - device: Device to run the model on (e.g., 'cuda' or 'cpu')

        Returns:
        - predicted_classes: Tensor of predicted class indices
        - probabilities: Tensor of class probabilities
        """
        self.to(self.device)
        self.eval()
        with torch.no_grad():
            # Get image embeddings
            image_embeddings = self.base_model.visual_projection(self.base_model.vision_model(images.to(self.device)).pooler_output)
            
            # Get text embeddings for all classes
            text_inputs = self.processor(text=class_names, return_tensors="pt", padding=True, truncation=True).to(self.device)
            text_embeddings = self.base_model.text_projection(self.base_model.text_model(**text_inputs).pooler_output)

            # Normalize embeddings
            image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
            text_embeddings = F.normalize(text_embeddings, p=2, dim=1)

            # Compute similarity scores
            similarity_scores = torch.matmul(image_embeddings, text_embeddings.t())

            # Get probabilities
            probabilities = F.softmax(similarity_scores, dim=1)

            # Get predicted classes
            predicted_classes = torch.argmax(probabilities, dim=1)

        return [label for label in predicted_classes], probabilities


    def train_model(self, train_dataloader, val_dataloader, test_dataloader, num_epochs=30, fixed_lr=False, learning_rate=1e-5, patience=5, max_len=512):
        self.to(self.device)

        best_val_loss = float('inf')
        best_val_f1 = 0.
        best_val_acc = 0.
        epochs_without_improvement = 0

        for param in self.parameters():
            param.requires_grad = True

        if(self.frozen_backbone=='not_last_layer'):
            for name, param in self.base_model.named_parameters():
                if ("visual_proj" in name) or ("text_proj" in name):
                    print(name)
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
                if "vision_model" in name:
                    param.requires_grad = False

        #criterion = nn.BCEWithLogitsLoss()
        criterion = nn.CrossEntropyLoss()

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
            # Add classifier parameters if any require gradients
            classifier_params = [p for p in self.classifier.parameters() if p.requires_grad]
            if classifier_params:
                param_groups.append({'params': classifier_params, 'lr': 1e-1})

            optimizer = torch.optim.AdamW(param_groups)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs-1, eta_min=1e-5)
        
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'train_f1': [], 'val_f1': [],
                   'train_class_f1': [], 'val_class_f1': []}
        
        for epoch in range(num_epochs):
            self.train()
            train_loss = 0.0
            all_train_predictions = []
            all_train_labels = []

            train_loop = tqdm(enumerate(train_dataloader), desc=f"Epoch {epoch+1}/{num_epochs} [Train] (lr={[group['lr'] for group in optimizer.param_groups]})")
            for batch_idx,batch in train_loop:

                labels = batch['label'].to(self.device)
                
                if(self.combine_type=='text'):
                    texts = batch['text']
                    images = torch.zeros(len(texts), 3, 224, 224, dtype=torch.float).to(self.device)
                elif(self.combine_type=='image'):
                    images = batch['image'].to(self.device)
                    texts = [""] * len(images)
                else:
                    images = batch['image'].to(self.device)
                    texts = batch['text']

                images = images.to(self.device)
                inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)

                optimizer.zero_grad()
                logits = self(input_ids=inputs.input_ids, 
                                attention_mask=inputs.attention_mask, 
                                pixel_values=images)  # use raw images because already transformed in dataloader

                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                probabilities = torch.nn.functional.softmax(logits, dim=1)
                predictions = logits.argmax(dim=1)

                train_loss += loss.item()
                all_train_predictions.extend(predictions.cpu().numpy())
                all_train_labels.extend(labels.cpu().numpy())

                train_loop.set_postfix(loss=loss.item())

            # Calculate training metrics
            train_loss /= len(train_dataloader)
            all_train_predictions = np.array(all_train_predictions)
            all_train_labels = np.array(all_train_labels)
            train_accuracy = accuracy_score(all_train_labels.flatten(), all_train_predictions.flatten())
            train_f1 = f1_score(all_train_labels, all_train_predictions, average='macro')

            # Per-class metrics
            train_class_f1 = dict(zip(self.class_dict.values(),f1_score(all_train_labels, all_train_predictions, average=None)))

            # Validation phase
            val_loss, val_accuracy, val_f1, val_class_f1 = self.evaluate(val_dataloader)

            # Record metrics
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_accuracy)
            history['train_f1'].append(train_f1)
            history['train_class_f1'].append(train_class_f1)

            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_accuracy)
            history['val_f1'].append(val_f1)
            history['val_class_f1'].append(val_class_f1)

            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Train F1: {train_f1:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val F1: {val_f1:.4f}")

            # Early stopping check
            #if val_loss < best_val_loss:
            if(val_f1 > best_val_f1):
                #best_val_loss = val_loss
                best_val_f1 = val_f1
                epochs_without_improvement = 0
                best_model_state = copy.deepcopy(self.state_dict())
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            scheduler.step()

        self.load_state_dict(best_model_state)

        test_loss, test_accuracy, test_f1, test_class_f1 = self.evaluate(test_dataloader)

        history['test_loss'] = test_loss
        history['test_acc'] = test_accuracy
        history['test_f1']= test_f1
        history['test_class_f1'] = test_class_f1

        best_epoch = int(np.argmax(history['val_f1']))
        print(f"\n=== Best Epoch: {best_epoch+1} (val_f1 = {history['val_f1'][best_epoch]:.4f}) ===")

        # Display metrics
        print(f"Train Acc: {history['train_acc'][best_epoch]:.4f}, Val Acc: {history['val_acc'][best_epoch]:.4f}, Test Acc: {history['test_acc']:.4f}")
        print(f"Train F1: {history['train_f1'][best_epoch]:.4f}, Val F1: {history['val_f1'][best_epoch]:.4f}, Test F1: {history['test_f1']:.4f} ")

        return history, best_model_state

    def evaluate(self, dataloader):
        self.eval()
        val_loss = 0.0
        all_predictions, all_labels = [], []
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch in dataloader:

                labels = batch['label']

                if(self.combine_type=='text'):
                    texts = batch['text']
                    images = torch.zeros(len(texts), 3, 224, 224, dtype=torch.float).to(self.device)
                elif(self.combine_type=='image'):
                    images = batch['image'].to(self.device)
                    texts = [""] * len(images)
                else:
                    images = batch['image'].to(self.device)
                    texts = batch['text']

                label_indices = torch.tensor([label for label in labels]).to(self.device)
                images = images.to(self.device)
                inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)

                logits = self(input_ids=inputs.input_ids, 
                                attention_mask=inputs.attention_mask, 
                                pixel_values=images) # use raw images because already transformed in dataloader
                
                loss = criterion(logits, label_indices)

                predictions = logits.argmax(dim=1)
                
                val_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(label_indices.cpu().numpy())

        val_loss /= len(dataloader)
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        accuracy = accuracy_score(all_labels.flatten(), all_predictions.flatten())
        f1 = f1_score(all_labels, all_predictions, average='macro')

        # Per-class metrics
        class_f1 = dict(zip(self.class_dict.values(),f1_score(all_labels, all_predictions, average=None)))

        return val_loss, accuracy, f1, class_f1
    
    def contrastive_train(self, train_dataloader, val_dataloader, test_dataloader, num_epochs=10, learning_rate=1e-5, patience=5, temperature=0.07, on_concept=False, concept_list=[]):
        self.to(self.device)

        best_val_loss = float('inf')
        best_val_f1 = 0.
        epochs_without_improvement = 0

        for param in self.parameters():
            param.requires_grad = True

        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs-1, eta_min=learning_rate/10)
        
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'train_f1': [], 'val_f1': [], 'train_class_f1': [], 'val_class_f1': []}

        for epoch in range(num_epochs):
            self.train()
            train_loss = 0.0
            train_logits = []
            train_preds = []
            train_true = []

            train_loop = tqdm(enumerate(train_dataloader), desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
            for batch_idx, batch in train_loop:
                
                if(self.combine_type=='image'):
                    images = batch['image'].to(self.device)
                    image_embeddings = self.base_model.visual_projection(self.base_model.vision_model(images).pooler_output)
                    image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
                elif(self.combine_type=='text'):
                    texts = batch['text']
                    text_inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
                    text_embeddings = self.base_model.text_projection(self.base_model.text_model(**text_inputs).pooler_output)
                    text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
                else:
                    images = batch['image'].to(self.device)
                    image_embeddings = self.base_model.visual_projection(self.base_model.vision_model(images).pooler_output)
                    image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
                    texts = batch['text']
                    text_inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
                    text_embeddings = self.base_model.text_projection(self.base_model.text_model(**text_inputs).pooler_output)
                    text_embeddings = F.normalize(text_embeddings, p=2, dim=1)

                # labels
                text_labels = [self.class_dict[i] for i in range(len(self.class_dict))]

                labels = batch['label'].to(self.device)
                #oh_labels = F.one_hot(labels, num_classes=len(self.class_dict.values())).float()

                if(on_concept):
                    # text labels corresponding to concept names
                    text_labels = concept_list
                    labels = torch.stack([batch[col] for col in concept_list], dim=1).to(self.device)  # Shape: [batch_size, n_concepts]

                # Convert text labels
                text_labels = [re.sub(r"[^a-zA-Z\s]", "", label.replace('image_', '').replace('text_', '').replace('concept_', '').replace('_', ' ').replace('::', ' ')) for label in text_labels]
                text_inputs = self.processor(text=text_labels, return_tensors="pt", padding=True, truncation=True).to(self.device)
                comparison_embeddings = self.base_model.text_projection(self.base_model.text_model(**text_inputs).pooler_output)
                comparison_embeddings = F.normalize(comparison_embeddings, p=2, dim=1)

                # Compute similarity scores
                if(self.combine_type=='image'):
                    logits = torch.matmul(image_embeddings, comparison_embeddings.t()) / temperature
                elif(self.combine_type=='text'):
                    logits = torch.matmul(text_embeddings, comparison_embeddings.t()) / temperature
                else:
                    logits_images = torch.matmul(image_embeddings, comparison_embeddings.t()) / temperature
                    logits_texts = torch.matmul(text_embeddings, comparison_embeddings.t()) / temperature

                    concept_combined_probs = torch.sigmoid(logits_images) + torch.sigmoid(logits_texts) - torch.sigmoid(logits_images) * torch.sigmoid(logits_texts) # element wise product
                    logits = torch.logit(concept_combined_probs.clamp(min=1e-7, max=1-1e-7))

                # BCEwithLogitsLoss applies internally softmax to logits which will give a probability from 0 to 1 for each class label
                # the goal is that the probability of the true label is higher than the probability of other labels
                if(not on_concept):
                    loss = F.cross_entropy(logits, labels)
                    #labels_flat = torch.t(oh_labels).contiguous().view(-1).float()  # Shape: [n_concepts*batch_size]
                    #logits_flat = logits.transpose(0, 1).contiguous().view(-1).float()  # Shape: [n_concepts*batch_size]
                    #loss = nn.BCEWithLogitsLoss()(logits_flat, labels_flat)
                else:
                    labels_flat = torch.t(labels).contiguous().view(-1).float()  # Shape: [n_concepts*batch_size]
                    logits_flat = torch.t(logits).contiguous().view(-1).float()  # Shape: [n_concepts*batch_size]
                    loss = nn.BCEWithLogitsLoss()(logits_flat, labels_flat)

                if(epoch>0):
                # for epoch=0, just take base_model without training
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                train_loss += loss.item()
                train_loop.set_postfix(loss=loss.item())

                # Collect predictions and true labels for metrics
                if(not on_concept):
                    predicted = torch.argmax(logits, dim=1)
                else:
                    predicted = (logits > 0.).float()

                train_logits.extend(logits.cpu().tolist())
                train_preds.extend(predicted.cpu().tolist())
                train_true.extend(labels.cpu().tolist())

            scheduler.step()

            train_loss /= len(train_dataloader)
            train_true = np.array(train_true)
            train_preds = np.array(train_preds)
            train_acc = accuracy_score(train_true.flatten(), train_preds.flatten())
            train_f1 = f1_score(train_true, train_preds, average='macro')

            # Per-class metrics
            if(on_concept==False):
                train_class_f1 = dict(zip(self.class_dict.values(),f1_score(train_true, train_preds, average=None)))
            else:
                # Calculate per-concept F1 scores
                train_class_f1 = dict(zip(concept_list,f1_score(train_true, train_preds, average=None)))

            # Validation phase
            val_loss, val_acc, val_f1, val_class_f1 = self.contrastive_evaluate(val_dataloader, temperature, on_concept=on_concept, concept_list=concept_list)

            # Record metrics
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['train_f1'].append(train_f1)
            history['train_class_f1'].append(train_class_f1)

            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)
            history['val_class_f1'].append(val_class_f1)

            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

            # Early stopping check
            #if val_loss < best_val_loss:
            if(val_f1 > best_val_f1):
                #best_val_loss = val_loss
                best_val_f1 = val_f1
                epochs_without_improvement = 0
                best_model_state = self.state_dict()
            else:
                epochs_without_improvement += 1

            if(val_f1 > best_val_f1):
                best_val_f1 = val_f1
                best_model_state = self.state_dict()

            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

        self.load_state_dict(best_model_state)

        test_loss, test_accuracy, test_f1, test_class_f1 = self.contrastive_evaluate(test_dataloader, temperature, on_concept=on_concept, concept_list=concept_list)

        history['test_loss'] = test_loss
        history['test_acc'] = test_accuracy
        history['test_f1']= test_f1
        history['test_class_f1'] = test_class_f1

        best_epoch = int(np.argmax(history['val_f1']))
        print(f"\n=== Best Epoch: {best_epoch+1} (val_f1 = {history['val_f1'][best_epoch]:.4f}) ===")

        # Display metrics
        print(f"Train Acc: {history['train_acc'][best_epoch]:.4f}, Val Acc: {history['val_acc'][best_epoch]:.4f}, Test Acc: {history['test_acc']:.4f}")
        print(f"Train F1: {history['train_f1'][best_epoch]:.4f}, Val F1: {history['val_f1'][best_epoch]:.4f}, Test F1: {history['test_f1']:.4f} ")

        return history, best_model_state

    def contrastive_evaluate(self, dataloader, temperature=0.07, on_concept=False, concept_list=[]):
        self.eval()
        val_loss = 0.0
        val_preds = []
        val_true = []

        with torch.no_grad():
            for batch in dataloader:

                if(self.combine_type=='image'):
                    images = batch['image'].to(self.device)
                    image_embeddings = self.base_model.visual_projection(self.base_model.vision_model(images).pooler_output)
                    image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
                elif(self.combine_type=='text'):
                    texts = batch['text']
                    text_inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
                    text_embeddings = self.base_model.text_projection(self.base_model.text_model(**text_inputs).pooler_output)
                    text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
                else:
                    images = batch['image'].to(self.device)
                    image_embeddings = self.base_model.visual_projection(self.base_model.vision_model(images).pooler_output)
                    image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
                    texts = batch['text']
                    text_inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
                    text_embeddings = self.base_model.text_projection(self.base_model.text_model(**text_inputs).pooler_output)
                    text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
                
                text_labels = [self.class_dict[i] for i in range(len(self.class_dict))]
                labels = batch['label'].to(self.device)
                #oh_labels = F.one_hot(labels, num_classes=len(self.class_dict.values())).float()

                if(on_concept):
                    # text labels corresponding to concept names
                    text_labels = concept_list
                    labels = torch.stack([batch[col] for col in concept_list], dim=1).to(self.device)  # Shape: [batch_size, n_concepts]

                # Convert text labels
                text_labels = [re.sub(r"[^a-zA-Z\s]", "", label.replace('image_', '').replace('text_', '').replace('concept_', '').replace('_', ' ').replace('::', ' ')) for label in text_labels]
                text_inputs = self.processor(text=text_labels, return_tensors="pt", padding=True, truncation=True).to(self.device)
                comparison_embeddings = self.base_model.text_projection(self.base_model.text_model(**text_inputs).pooler_output)
                comparison_embeddings = F.normalize(comparison_embeddings, p=2, dim=1)

                # Compute similarity scores
                if(self.combine_type=='image'):
                    logits = torch.matmul(image_embeddings, comparison_embeddings.t()) / temperature
                elif(self.combine_type=='text'):
                    logits = torch.matmul(text_embeddings, comparison_embeddings.t()) / temperature
                else:
                    logits_images = torch.matmul(image_embeddings, comparison_embeddings.t()) / temperature
                    logits_texts = torch.matmul(text_embeddings, comparison_embeddings.t()) / temperature

                    concept_combined_probs = torch.sigmoid(logits_images) + torch.sigmoid(logits_texts) - torch.sigmoid(logits_images) * torch.sigmoid(logits_texts) # element wise product
                    logits = torch.logit(concept_combined_probs.clamp(min=1e-7, max=1-1e-7))

                # BCEwithLogitsLoss applies internally softmax to logits which will give a probability from 0 to 1 for each class label
                # the goal is that the probability of the true label is higher than the probability of other labels
                if(not on_concept):
                    loss = F.cross_entropy(logits, labels)
                else:
                    labels_flat = torch.t(labels).contiguous().view(-1).float()  # Shape: [n_concepts*batch_size]
                    logits_flat = torch.t(logits).contiguous().view(-1).float()  # Shape: [n_concepts*batch_size]
                    loss = nn.BCEWithLogitsLoss()(logits_flat, labels_flat)

                val_loss += loss.item()

                # Collect predictions and true labels for metrics
                if(not on_concept):
                    predicted = torch.argmax(logits, dim=1)
                else:
                    predicted = (logits > 0.).float()

                val_preds.extend(predicted.cpu().tolist())
                val_true.extend(labels.cpu().tolist())

        val_loss /= len(dataloader)
        val_true = np.array(val_true)
        val_preds = np.array(val_preds)
        val_acc = accuracy_score(val_true.flatten(), val_preds.flatten())
        val_f1 = f1_score(val_true, val_preds, average='macro')

        # Per-class metrics
        if(on_concept==False):
            val_class_f1 = dict(zip(self.class_dict.values(),f1_score(val_true, val_preds, average=None)))
        else:
            # Calculate per-concept F1 scores
            val_class_f1 = dict(zip(concept_list,f1_score(val_true, val_preds, average=None)))

        return val_loss, val_acc, val_f1, val_class_f1

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

def plot_history(
    history,
    description="",
    show_task_f1: bool = True,
    training_mode=True,
    cbllm=False,
):
    import matplotlib.pyplot as plt
    import numpy as np

    has_concept_acc = 'train_concept_acc' in history and 'val_concept_acc' in history
    has_concept_f1 = 'train_concept_f1' in history and 'val_concept_f1' in history
    has_concept_counts = 'concept_counts' in history
    has_test_per_concept_f1 = 'test_per_concept_f1' in history
    has_vif = 'vif_dict' in history
    has_coeff = 'coeff_dict' in history
    has_mi = 'mi_dict' in history

    num_plots = 0
    if(training_mode):
        num_plots = 2
        if show_task_f1:
            num_plots += 1
        if has_concept_acc or has_concept_f1:
            num_plots += 1
        x_epochs = [x + 1 for x in range(len(history['train_loss']))]

    num_plots += 1 # per-class scores
    if(has_test_per_concept_f1):
        num_plots += 3 # concept scores
    if(has_concept_counts):
        num_plots += 2 # concept scores
    if(has_vif):
        num_plots += 1 # vif scores
    if(has_coeff):
        num_plots += 1 # coeff scores
    if(has_mi):
        num_plots += 1 #mutual info
    
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 5 * num_plots))
    if num_plots == 1:
        axes = [axes]

    plot_idx = 0

    # scores for each epoch (available only when having just trained the model)
    if(training_mode):
        # Loss plot
        ax1 = axes[plot_idx]
        plot_idx += 1
        line1, = ax1.plot(x_epochs, history['train_loss'], label='Train Loss')
        #line2, = ax1.plot(x_epochs, history['val_loss'], label='Validation Loss')
        # Annotate each point
        for x, y in zip(x_epochs, history['train_loss']):
            ax1.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8, color=line1.get_color())
        #for x, y in zip(x_epochs, history['val_loss']):
        #    ax1.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0,-10), ha='center', fontsize=8, color=line2.get_color())
        ax1.set_title(f'Model Loss {description}')
        ax1.set_ylabel('Loss')
        ax1.set_xlabel('Epoch')
        ax1.legend()

        # Accuracy plot
        ax2 = axes[plot_idx]
        plot_idx += 1
        line1, = ax2.plot(x_epochs, history['train_acc'], label='Train Accuracy')
        line2, = ax2.plot(x_epochs, history['val_acc'], label='Validation Accuracy')
        for x, y in zip(x_epochs, history['train_acc']):
            ax2.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8, color=line1.get_color())
        for x, y in zip(x_epochs, history['val_acc']):
            ax2.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0,-10), ha='center', fontsize=8, color=line2.get_color())
        ax2.set_title(f'Model Accuracy {description}')
        ax2.set_ylabel('Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylim(0, 1)
        ax2.legend()

        # Optionally plot task F1
        if show_task_f1:
            ax3 = axes[plot_idx]
            plot_idx += 1
            line1, = ax3.plot(x_epochs, history['train_f1'], label='Train F1')
            line2, = ax3.plot(x_epochs, history['val_f1'], label='Validation F1')
            for x, y in zip(x_epochs, history['train_f1']):
                ax3.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8, color=line1.get_color())
            for x, y in zip(x_epochs, history['val_f1']):
                ax3.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0,-10), ha='center', fontsize=8, color=line2.get_color())
            ax3.set_title(f'Model F1 Score {description}')
            ax3.set_ylabel('F1 Score')
            ax3.set_xlabel('Epoch')
            ax3.set_ylim(0, 1)
            ax3.legend()

        # Combined Concept Accuracy and Concept F1 plot
        if has_concept_acc or has_concept_f1:
            ax_concept = axes[plot_idx]
            plot_idx += 1
            if has_concept_acc:
                line1, = ax_concept.plot(x_epochs, history['train_concept_acc'], label='Train Concept Accuracy' if not cbllm else 'Train Concept RMSE', linestyle='--')
                line2, = ax_concept.plot(x_epochs, history['val_concept_acc'], label='Val Concept Accuracy' if not cbllm else 'Val Concept RMSE', linestyle='--')
                for x, y in zip(x_epochs, history['train_concept_acc']):
                    ax_concept.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8, color=line1.get_color())
                for x, y in zip(x_epochs, history['val_concept_acc']):
                    ax_concept.annotate(f"{float(y):.3f}", (x, y), textcoords="offset points", xytext=(0,-10), ha='center', fontsize=8, color=line2.get_color())
            if has_concept_f1:
                line3, = ax_concept.plot(x_epochs, history['train_concept_f1'], label='Train Concept F1' if not cbllm else 'Train Concept Cosine Similarity')
                line4, = ax_concept.plot(x_epochs, history['val_concept_f1'], label='Val Concept F1' if not cbllm else 'Val Concept Cosine Similarity')
                for x, y in zip(x_epochs, history['train_concept_f1']):
                    ax_concept.annotate(f"{float(y):.3f}", (x, y), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8, color=line3.get_color())
                for x, y in zip(x_epochs, history['val_concept_f1']):
                    ax_concept.annotate(f"{float(y):.3f}", (x, y), textcoords="offset points", xytext=(0,-10), ha='center', fontsize=8, color=line4.get_color())
            ax_concept.set_title(f'Concept Accuracy & Concept F1 {description}')
            ax_concept.set_ylabel('Score')
            ax_concept.set_xlabel('Epoch')
            ax_concept.set_ylim(0, 1)
            ax_concept.legend()

    # Class-wise F1 (bar, last epoch)
    ax4 = axes[plot_idx]
    plot_idx += 1
    labels = []
    if(training_mode):
        dict_train_class = history['train_class_f1'][-1]
        dict_val_class = history['val_class_f1'][-1]
        labels += list(dict_train_class.keys())
        labels += list(dict_val_class.keys())
    dict_test_class = history['test_class_f1']
    labels += list(dict_test_class.keys())
    labels = sorted(np.unique(labels), key=lambda x: dict_test_class[x], reverse=True)
    if(training_mode):
        train_scores = [dict_train_class[label] for label in labels]
        val_scores = [dict_val_class[label] for label in labels]
    test_scores = [dict_test_class[label] for label in labels]
    bar_width = 0.35
    x = np.arange(len(labels))
    if(training_mode):
        bars1 = ax4.bar(x, train_scores, width=bar_width, label='Train F1', alpha=0.7)
        bars2 = ax4.bar(x + bar_width, val_scores, width=bar_width, label='Validation F1', alpha=0.7)
        bars3 = ax4.bar(x + 2*bar_width, test_scores, width=bar_width, label='Test F1', alpha=0.7)
    else:
        bars3 = ax4.bar(x, test_scores, width=bar_width, label='Test F1', alpha=0.7)
    # Annotate bars
    if(training_mode):
        for bar in bars1:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2, height, f"{height:.2f}", ha='center', va='bottom', fontsize=8, color=bar.get_facecolor())
        for bar in bars2:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2, height, f"{height:.2f}", ha='center', va='bottom', fontsize=8, color=bar.get_facecolor())
    for bar in bars3:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2, height, f"{height:.2f}", ha='center', va='bottom', fontsize=8, color=bar.get_facecolor())
    ax4.set_title(f'Class-wise F1 Scores (Last Epoch) {description}')
    ax4.set_xlabel('Labels')
    ax4.set_ylabel('F1 Score')
    ax4.set_xticks(x + bar_width / 2)
    ax4.set_xticklabels(labels, rotation=90)
    ax4.set_ylim(0, 1)
    ax4.legend()

    # Per-concept F1 (bar, last epoch)
    if(has_test_per_concept_f1):
        ax_per_concept = axes[plot_idx]
        plot_idx += 1
        if(training_mode):
            train_per_concept = history['train_per_concept_f1'][-1]
            val_per_concept = history['val_per_concept_f1'][-1]
        test_per_concept = history['test_per_concept_f1']
        N = 15
        concepts = sorted(list(test_per_concept.keys()), key=lambda x: test_per_concept[x], reverse=True)[:N]
        if(training_mode):
            concept_train_scores = [train_per_concept.get(concept, 0) for concept in concepts]
            concept_val_scores = [val_per_concept.get(concept, 0) for concept in concepts]
        concept_test_scores = [test_per_concept.get(concept, 0) for concept in concepts]
        bar_width = 0.35
        x = np.arange(len(concepts))
        if(training_mode):
            bars1 = ax_per_concept.bar(x, concept_train_scores, width=bar_width, label='Train F1' if not cbllm else 'Train Concept Cosine Similarity', alpha=0.7)
            bars2 = ax_per_concept.bar(x + bar_width, concept_val_scores, width=bar_width, label='Validation F1' if not cbllm else 'Validation Concept Cosine Similarity', alpha=0.7)
            bars3 = ax_per_concept.bar(x + 2*bar_width, concept_test_scores, width=bar_width, label='Test F1' if not cbllm else 'Test Concept Cosine Similarity', alpha=0.7)
        else:
            bars3 = ax_per_concept.bar(x, concept_test_scores, width=bar_width, label='Test F1', alpha=0.7)
        # Annotate bars
        if(training_mode):
            for bar in bars1:
                height = bar.get_height()
                ax_per_concept.text(bar.get_x() + bar.get_width()/2, height, f"{height:.2f}", ha='center', va='bottom', fontsize=8, color=bar.get_facecolor())
            for bar in bars2:
                height = bar.get_height()
                ax_per_concept.text(bar.get_x() + bar.get_width()/2, height, f"{height:.2f}", ha='center', va='bottom', fontsize=8, color=bar.get_facecolor())
        for bar in bars3:
            height = bar.get_height()
            ax_per_concept.text(bar.get_x() + bar.get_width()/2, height, f"{height:.2f}", ha='center', va='bottom', fontsize=8, color=bar.get_facecolor())
        ax_per_concept.set_title(f'Top {N} Concept-wise F1 Scores (Last Epoch) {description}')
        ax_per_concept.set_xlabel('Concepts')
        ax_per_concept.set_ylabel('F1 Score')
        ax_per_concept.set_xticks(x + bar_width / 2)
        shortened_concepts = [c.replace('concept_','')[:20] + '...' if len(c) > 30 else c for c in concepts]
        ax_per_concept.set_xticklabels(shortened_concepts, rotation=90)
        ax_per_concept.set_ylim(0, 1)
        ax_per_concept.legend()

        # leakage vs accuracy
        ax_leak_acc = axes[plot_idx]
        plot_idx += 1
        score = list(history['test_per_concept_acc'].values())
        leakage = list(history['test_per_concept_leakage'].values())
        concept_list = list(history['test_per_concept_acc'].keys())
        for i, concept in enumerate(concept_list):
            ax_leak_acc.scatter(score[i], leakage[i], label=concept)
            ax_leak_acc.text(score[i], leakage[i], concept, fontsize=8)
        ax_leak_acc.set_xlabel('Concept accuracy' if not cbllm else 'Concept RMSE')
        ax_leak_acc.set_ylabel('Leakage Score')

        # leakage vs f1 score
        ax_leak_f1 = axes[plot_idx]
        plot_idx += 1
        score = list(history['test_per_concept_f1'].values())
        leakage = list(history['test_per_concept_leakage'].values())
        concept_list = list(history['test_per_concept_f1'].keys())
        for i, concept in enumerate(concept_list):
            ax_leak_f1.scatter(score[i], leakage[i], label=concept)
            ax_leak_f1.text(score[i], leakage[i], concept, fontsize=8)
        ax_leak_f1.set_xlabel('Concept f1-score' if not cbllm else 'Concept Cosine Similarity')
        ax_leak_f1.set_ylabel('Leakage Score')

    if(has_concept_counts):
        ax5 = axes[plot_idx]
        plot_idx += 1
        concept_list = list(history['concept_counts'].keys())
        concept_list = [c for c in concept_list if not c.endswith('_classactivations')]
        freq = [history['concept_counts'][concept] for concept in concept_list]
        leakage = [history['test_per_concept_leakage'][concept] for concept in concept_list]
        for i, concept in enumerate(concept_list):
            ax5.scatter(freq[i], leakage[i], label=concept)
            ax5.text(freq[i], leakage[i], concept, fontsize=8)
            #ax5.annotate(f"{leakage[i]:.2f}", (freq[i], leakage[i]), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8)
        ax5.set_xlabel('Concept Frequency')
        ax5.set_ylabel('Leakage Score')

        ax6 = axes[plot_idx]
        plot_idx += 1
        concept_otherclass_list = [c for c in history['concept_counts'].keys() if c.endswith('_classactivations')]
        concept_otherclass_base = [c.replace('_classactivations', '') for c in concept_otherclass_list]
        freq_otherclass = [history['concept_counts'][concept] for concept in concept_otherclass_list]
        leakage_otherclass = [history['test_per_concept_leakage'][concept] for concept in concept_otherclass_base]
        for i, concept in enumerate(concept_otherclass_base):
            ax6.scatter(freq_otherclass[i], leakage_otherclass[i], label=concept)
            ax6.text(freq_otherclass[i], leakage_otherclass[i], concept, fontsize=8)
            #ax6.annotate(f"{leakage_otherclass[i]:.2f}", (freq_otherclass[i], leakage_otherclass[i]), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8)
        ax6.set_xlabel('# of Class Activations')
        ax6.set_ylabel('Leakage Score')

    if(has_vif):
        ax6 = axes[plot_idx]
        plot_idx += 1
        concept_list = list(history['vif_dict'].keys())
        freq = [history['vif_dict'][concept] for concept in concept_list]
        leakage = [history['test_per_concept_leakage'][concept] for concept in concept_list]
        for i, concept in enumerate(concept_list):
            ax6.scatter(freq[i], leakage[i], label=concept)
            ax6.text(freq[i], leakage[i], concept, fontsize=8)
        ax6.set_xlabel('VIF')
        ax6.set_ylabel('Leakage Score')

    if(has_coeff):
        ax7 = axes[plot_idx]
        plot_idx += 1
        concept_list = list(history['coeff_dict'].keys())
        freq = [history['coeff_dict'][concept] for concept in concept_list]
        leakage = [history['test_per_concept_leakage'][concept] for concept in concept_list]
        for i, concept in enumerate(concept_list):
            ax7.scatter(freq[i], leakage[i], label=concept)
            ax7.text(freq[i], leakage[i], concept, fontsize=8) 
        ax7.set_xlabel('Regression coefficient (independant)')
        ax7.set_ylabel('Leakage Score')

    if(has_mi):
        ax8 = axes[plot_idx]
        plot_idx += 1
        concept_list = list(history['mi_dict'].keys())
        freq = [history['mi_dict'][concept] for concept in concept_list]
        leakage = [history['test_per_concept_leakage'][concept] for concept in concept_list]
        for i, concept in enumerate(concept_list):
            ax8.scatter(freq[i], leakage[i], label=concept)
            ax8.text(freq[i], leakage[i], concept, fontsize=8)
        ax8.set_xlabel('Mutual Information between concepts and target')
        ax8.set_ylabel('Leakage Score')

    plt.tight_layout()
    plt.show()
   