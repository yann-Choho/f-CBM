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

from transformers import CLIPModel, CLIPProcessor, BlipModel, BlipProcessor
from models.utils import ElasticNetLinearLayer

class JointModel:
    def __init__(
        self,
        backbone: str,  # 'clip' ou 'blip'
        ModelXtoCtoY_layer,
        config,
        train_loader,
        val_loader,
        use_cls_token: bool = True,
        num_discovered_concepts: int = 1,
        linear_layer=None
    ):
        # Mode image seulement - pas de features multimodales
        self.image_only_features = True
       
        # Initialize encoder and processor based on chosen backbone - IMAGE ONLY
        if backbone == 'clip':
            self.embedder_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.tokenizer = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self._is_clip = True
        elif backbone == 'blip':
            self.embedder_model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
            self.tokenizer = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self._is_clip = False
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.ModelXtoCtoY_layer = ModelXtoCtoY_layer
        self.config = config
        self.device = config.device
        self.model_name = config.model_name
        self.num_epochs = config.num_epochs
        self.lambda_XtoC = config.lambda_XtoC

        self.use_cls_token = use_cls_token
        self.linear_layer = linear_layer

        self.optimizer = Adam(
            list(self.embedder_model.parameters()) + list(self.ModelXtoCtoY_layer.parameters()),
            lr=1e-5)

        self.loss_concept_function = BCEWithLogitsLoss()
        self.loss_fn = CrossEntropyLoss()
       
        self.embedder_model.to(self.device)
        self.ModelXtoCtoY_layer.to(self.device)
        if self.linear_layer is not None:
            self.linear_layer.to(self.device)
                    
        self.concepts_name = [] 
        self.alternate_save = False
        self.strategy = None # 'random', 'tcavs', 'lig', 'frequences'

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

    def get_XtoY_output(self, pooled_output):
        """
        Just a useful function to get the output of the XtoY layer
        """
        outputs = self.ModelXtoCtoY_layer(pooled_output)
        XtoC_output = outputs[1:] 
        XtoY_output = outputs[0:1]
        return XtoY_output, XtoC_output

    def forward(self, batch):
        """
        Forward pass: encode image only, then predict concepts and task label.
        """
        # Encode images only - no text
        images = batch["image"].to(self.device)
        pooled_output = self.get_pooled_output(images)
        
        outputs = self.ModelXtoCtoY_layer(pooled_output)
        XtoY_output = outputs[0:1] 
        XtoC_output = outputs[1:]
        
        return XtoY_output, XtoC_output, pooled_output

    def train_model(self, train_loader, val_loader):
        """Train the joint model using images only"""
        print("Starting training with image-only features...")
        
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        for epoch in range(self.num_epochs):
            # Training phase
            self.embedder_model.train()
            self.ModelXtoCtoY_layer.train()
            
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                
                # Collect concept labels
                concept_labels = []
                for key in batch.keys():
                    if key.startswith("concept_"):
                        concept_labels.append(batch[key].float().to(self.device))
                
                self.optimizer.zero_grad()
                
                # Forward pass - image only
                pooled_output = self.get_pooled_output(images)
                outputs = self.ModelXtoCtoY_layer(pooled_output)
                
                XtoY_output = outputs[0:1].squeeze()  # Task prediction
                XtoC_outputs = outputs[1:]  # Concept predictions
                
                # Task loss
                task_loss = self.loss_fn(XtoY_output, labels)
                
                # Concept loss
                concept_loss = 0.0
                if len(concept_labels) > 0 and len(XtoC_outputs) > 0:
                    for i, concept_label in enumerate(concept_labels):
                        if i < len(XtoC_outputs):
                            concept_loss += self.loss_concept_function(
                                XtoC_outputs[i].squeeze(), concept_label
                            )
                
                # Combined loss
                total_loss = task_loss + self.lambda_XtoC * concept_loss
                total_loss.backward()
                self.optimizer.step()
                
                train_loss += total_loss.item()
                _, predicted = torch.max(XtoY_output.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                # Memory cleanup
                del images, labels, pooled_output, outputs
                torch.cuda.empty_cache()
            
            # Calculate training metrics
            train_accuracy = train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)
            
            self.train_losses.append(avg_train_loss)
            self.train_accuracies.append(train_accuracy)
            
            # Validation phase
            val_accuracy, avg_val_loss = self.evaluate_model(val_loader)
            self.val_losses.append(avg_val_loss)
            self.val_accuracies.append(val_accuracy)
            
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
            print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            
        print("Training completed!")

    def evaluate_model(self, dataloader):
        """Evaluate the model using images only"""
        self.embedder_model.eval()
        self.ModelXtoCtoY_layer.eval()
        
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                
                # Collect concept labels
                concept_labels = []
                for key in batch.keys():
                    if key.startswith("concept_"):
                        concept_labels.append(batch[key].float().to(self.device))
                
                # Forward pass - image only
                pooled_output = self.get_pooled_output(images)
                outputs = self.ModelXtoCtoY_layer(pooled_output)
                
                XtoY_output = outputs[0:1].squeeze()
                XtoC_outputs = outputs[1:]
                
                # Task loss
                task_loss = self.loss_fn(XtoY_output, labels)
                
                # Concept loss
                concept_loss = 0.0
                if len(concept_labels) > 0 and len(XtoC_outputs) > 0:
                    for i, concept_label in enumerate(concept_labels):
                        if i < len(XtoC_outputs):
                            concept_loss += self.loss_concept_function(
                                XtoC_outputs[i].squeeze(), concept_label
                            )
                
                total_loss += (task_loss + self.lambda_XtoC * concept_loss).item()
                
                _, predicted = torch.max(XtoY_output.data, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
                
                # Memory cleanup
                del images, labels, pooled_output, outputs
                torch.cuda.empty_cache()
        
        accuracy = correct_predictions / total_predictions
        avg_loss = total_loss / len(dataloader)
        
        return accuracy, avg_loss

    def save_model(self, save_path, suffix="image_only"):
        """Save the trained model"""
        os.makedirs(f"{save_path}/blue_checkpoints/{self.model_name}/JointModel", exist_ok=True)
        
        # Save embedder model
        embedder_path = f"{save_path}/blue_checkpoints/{self.model_name}/JointModel/{self.model_name}_embedder_state_dict_{suffix}.pth"
        torch.save(self.embedder_model.state_dict(), embedder_path)
        
        # Save joint layer
        joint_path = f"{save_path}/blue_checkpoints/{self.model_name}/JointModel/{self.model_name}_joint_layer_state_dict_{suffix}.pth"
        torch.save(self.ModelXtoCtoY_layer.state_dict(), joint_path)
        
        # Save linear layer if exists
        if self.linear_layer is not None:
            linear_path = f"{save_path}/blue_checkpoints/{self.model_name}/JointModel/{self.model_name}_linear_layer_state_dict_{suffix}.pth"
            torch.save(self.linear_layer.state_dict(), linear_path)
        
        print(f"Joint model saved to {save_path}")

    def load_model(self, save_path, suffix="image_only"):
        """Load the trained model"""
        # Load embedder model
        embedder_path = f"{save_path}/blue_checkpoints/{self.model_name}/JointModel/{self.model_name}_embedder_state_dict_{suffix}.pth"
        if os.path.exists(embedder_path):
            self.embedder_model.load_state_dict(torch.load(embedder_path, map_location=self.device))
            print(f"Embedder loaded from {embedder_path}")
        
        # Load joint layer
        joint_path = f"{save_path}/blue_checkpoints/{self.model_name}/JointModel/{self.model_name}_joint_layer_state_dict_{suffix}.pth"
        if os.path.exists(joint_path):
            self.ModelXtoCtoY_layer.load_state_dict(torch.load(joint_path, map_location=self.device))
            print(f"Joint layer loaded from {joint_path}")
        
        # Load linear layer if exists
        if self.linear_layer is not None:
            linear_path = f"{save_path}/blue_checkpoints/{self.model_name}/JointModel/{self.model_name}_linear_layer_state_dict_{suffix}.pth"
            if os.path.exists(linear_path):
                self.linear_layer.load_state_dict(torch.load(linear_path, map_location=self.device))
                print(f"Linear layer loaded from {linear_path}")

    def predict(self, dataloader):
        """Make predictions using images only"""
        self.embedder_model.eval()
        self.ModelXtoCtoY_layer.eval()
        
        all_predictions = []
        all_concept_predictions = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                images = batch["image"].to(self.device)
                
                # Forward pass - image only
                pooled_output = self.get_pooled_output(images)
                outputs = self.ModelXtoCtoY_layer(pooled_output)
                
                XtoY_output = outputs[0:1].squeeze()
                XtoC_outputs = outputs[1:]
                
                # Task predictions
                _, predicted = torch.max(XtoY_output.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                
                # Concept predictions
                batch_concept_preds = []
                for concept_output in XtoC_outputs:
                    concept_probs = torch.sigmoid(concept_output).cpu().numpy()
                    batch_concept_preds.append(concept_probs)
                
                if batch_concept_preds:
                    all_concept_predictions.append(np.array(batch_concept_preds))
                
                # Memory cleanup
                del images, pooled_output, outputs
                torch.cuda.empty_cache()
        
        return np.array(all_predictions), all_concept_predictions