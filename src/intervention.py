import numpy as np
import torch
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

class CBMInterventionEvaluator:
    """
    Handles intervention ordering computation and test-time intervention evaluation
    for PyTorch-based Concept Bottleneck Models.
    """
    def __init__(self, model):
        """
        Args:
            model: Your CBM model with classifier layer
        """
        self.model = model
        self.processor = self.model.processor
        self.device = self.model.device
        self.intervention_ordering = None
        self.concept_names = self.model.concept_list if hasattr(self.model, 'concept_list') else None
        self.model.eval()
    
    def get_predictions(self, dataloader, tau=0.5, intervened_concepts=None):
        """
        Get predictions from the model, optionally with interventions.
        
        Args:
            dataloader: DataLoader for the dataset
            tau: Temperature for concept binarization
            intervened_concepts: List of concept indices to intervene on (None = no intervention)
        
        Returns:
            concept_predictions: (N, n_concepts) predicted concepts
            task_predictions: (N,) predicted task labels
            true_labels: (N,) ground truth task labels
        """
        all_concept_preds = []
        all_task_preds = []
        all_true_labels = []
        all_true_concepts = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                labels = batch['label']
                
                if self.model.combine_type in ['concat', 'combine']:
                    texts = batch['text']
                    images = batch['image']
                elif self.model.combine_type == 'text':
                    texts = batch['text']
                    images = torch.zeros(len(texts), 3, 224, 224, dtype=torch.float).to(self.device)
                elif self.model.combine_type == 'image':
                    images = batch['image']
                    texts = [""] * len(images)
                
                label_indices = torch.tensor([label for label in labels]).to(self.device)
                images = images.to(self.device)
                inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
                
                # Forward to get predictions
                concept_logits, concept_repr, concept_repr_combined, _ = self.model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    pixel_values=images,
                    tau=tau
                )
                
                if(self.model.concept_representation!='importance'):
                    # Get binary concept predictions for C3M
                    concept_preds = self.model._get_binary_concepts(concept_logits, tau=tau)
                else:
                    # Take raw logits for CBLLM
                    concept_preds = concept_logits

                concept_labels = torch.stack([batch[col] for col in self.model.concept_list], dim=1).to(self.device).float()  # Shape: [batch_size, n_concepts]
                
                # Apply interventions if specified
                if intervened_concepts is not None:
                    # Replace predicted concepts with ground truth for intervened concepts
                    concept_preds_intervened = concept_preds.clone()
                    for concept_idx in intervened_concepts:
                        concept_preds_intervened[:, concept_idx] = concept_labels[:, concept_idx]
                    
                    concept_to_use = concept_preds_intervened
                else:
                    concept_to_use = concept_preds
                
                # Get task predictions from concepts using classifier
                logits = self.model.classifier(concept_to_use)
                task_preds = logits.argmax(dim=1)
                
                # Store results
                all_task_preds.extend(task_preds.cpu().numpy())
                all_true_labels.extend(label_indices.cpu().numpy())
                if intervened_concepts is not None:
                    all_concept_preds.extend(concept_to_use.cpu().numpy())
                else:
                    all_concept_preds.extend(concept_preds.cpu().numpy())
                
                # Store ground truth concepts
                all_true_concepts.extend(concept_labels.detach().cpu().numpy()) # Shape: [n_batch, n_concepts]
        
        task_predictions = np.array(all_task_preds)
        true_labels = np.array(all_true_labels)
        concept_predictions = np.array(all_concept_preds)
        concept_true = np.array(all_true_concepts)
        
        return concept_predictions, task_predictions, true_labels, concept_true
    
    def compute_intervention_ordering(self, val_dataloader, tau=0.5):
        """
        Compute input-independent intervention ordering using validation set.
        
        Args:
            val_dataloader: Validation DataLoader
            tau: Temperature for concept binarization
        
        Returns:
            List of concept indices sorted by intervention importance
        """
        print("Computing intervention ordering on validation set...")
        
        # Get baseline predictions and ground truth concepts
        C_pred, Y_pred_baseline, Y_true, C_true = self.get_predictions(val_dataloader, tau=tau)
        
        baseline_score = accuracy_score(Y_true, Y_pred_baseline)
        
        print(f"Baseline accuracy: {baseline_score:.4f}")
        print(f"  concept acc = {self._compute_concept_metrics(C_pred, C_true)}")
        print(f"Number of concepts: {len(self.model.concept_list)}")
        
        concept_improvements = []
        
        # Test single-concept intervention for each concept
        for j, concept_name in tqdm(enumerate(self.model.concept_list), desc="Testing concept interventions"):
            # Get predictions with intervention on concept j only
            C_pred_intervened, Y_pred_intervened, Y_true, C_true = self.get_predictions(
                val_dataloader, 
                tau=tau,
                intervened_concepts=[j]
            )
            
            intervened_score = accuracy_score(Y_true, Y_pred_intervened)
            improvement = intervened_score - baseline_score
            concept_improvements.append((concept_name, j, improvement))
            
            print(f"  {concept_name}: improvement = {improvement:.4f} (accuracy = {intervened_score:.4f})")
            print(f"  concept acc = {self._compute_concept_metrics(C_pred_intervened, C_true)}")
        
        # Sort by improvement (descending)
        concept_improvements.sort(key=lambda x: x[2], reverse=True)
        self.intervention_ordering = [idx for _, idx, _ in concept_improvements]
        
        print(f"\nIntervention ordering (by importance):")
        for rank, (concept_name, idx, imp) in enumerate(concept_improvements[:10], 1):  # Show top 10
            print(f"  {rank}. {concept_name}: +{imp:.4f}")
        
        return self.intervention_ordering
        
    def _compute_concept_metrics(self, C_pred, C_true):
        """Compute per-concept accuracy, acc/RMSE following your reference code pattern."""
        from sklearn.metrics import f1_score, accuracy_score, mean_squared_error
        
        per_concept_acc = {}
        
        for i, concept_name in enumerate(self.model.concept_list):
            concept_true = C_true[:, i]
            concept_pred = C_pred[:, i]
            
            if self.model.concept_representation != 'importance':
                # Binary classification metrics
                concept_acc = accuracy_score(concept_true, concept_pred)
            else:
                # Continuous metrics
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FutureWarning)
                    concept_acc = mean_squared_error(concept_true, concept_pred, squared=False)
            
            per_concept_acc[concept_name] = concept_acc
        
        return per_concept_acc

    def evaluate_interventions(self, test_dataloader, tau=0.5, max_interventions=None):
        """
        Evaluate test-time intervention with sequential concept interventions.
        
        Args:
            test_dataloader: Test DataLoader
            tau: Temperature for concept binarization
            max_interventions: Maximum number of concepts to intervene on (default: all)
        
        Returns:
            Dictionary with intervention results
        """
        if self.intervention_ordering is None:
            raise ValueError("Must compute intervention ordering first using compute_intervention_ordering()")
        
        print("\nEvaluating test-time interventions...")
        
        # Get baseline predictions and ground truth
        C_pred, Y_pred_baseline, Y_true, C_true = self.get_predictions(test_dataloader, tau=tau)
        
        if max_interventions is None:
            max_interventions = len(self.model.concept_list)
        
        results = {
            'num_interventions': [],
            'accuracy': [],
            'intervened_concepts': [],
            'intervened_concept_names': []
        }
        
        # Baseline (no intervention)
        baseline_acc = accuracy_score(Y_true, Y_pred_baseline)
        results['num_interventions'].append(0)
        results['accuracy'].append(baseline_acc)
        results['intervened_concepts'].append([])
        results['intervened_concept_names'].append([])
        print(f"k=0 interventions: accuracy = {baseline_acc:.4f}")
        print(f"  concept acc = {self._compute_concept_metrics(C_pred, C_true)}")
        
        # Sequential interventions
        intervened_so_far = []
        intervened_names_so_far = []
        
        for k in range(1, min(max_interventions + 1, len(self.model.concept_list) + 1)):
            # Get next concept to intervene on
            concept_idx = self.intervention_ordering[k - 1]
            intervened_so_far.append(concept_idx)
            
            concept_name = self.model.concept_list[concept_idx]
            intervened_names_so_far.append(concept_name)
            
            # Get predictions with k interventions
            C_pred_k, Y_pred_k, Y_true, C_true = self.get_predictions(
                test_dataloader,
                tau=tau,
                intervened_concepts=intervened_so_far
            )
            
            acc_k = accuracy_score(Y_true, Y_pred_k)
            
            results['num_interventions'].append(k)
            results['accuracy'].append(acc_k)
            results['intervened_concepts'].append(intervened_so_far.copy())
            results['intervened_concept_names'].append(intervened_names_so_far.copy())
            
            # Format concept names for printing
            if k <= 3:
                concepts_str = ", ".join(intervened_names_so_far)
            else:
                concepts_str = f"{intervened_names_so_far[0]}, ..., {concept_name}"
            
            print(f"k={k} interventions ({concepts_str}): accuracy = {acc_k:.4f}")
            print(f"  concept acc = {self._compute_concept_metrics(C_pred_k, C_true)}")
        
        return results
    
    def plot_intervention_curve(self, results, save_path='intervention_curve.png'):
        """Plot accuracy vs number of interventions with concept names."""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Main plot
        ax.plot(results['num_interventions'], results['accuracy'], 'o-', 
                linewidth=2, markersize=8, color='#2E86AB', label='Accuracy')
        
        ax.set_xlabel('Number of Concept Interventions', fontsize=13)
        ax.set_ylabel('Task Accuracy', fontsize=13)
        ax.set_title('Test-Time Intervention Performance', fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add improvement annotation
        if len(results['accuracy']) > 1:
            improvement = results['accuracy'][-1] - results['accuracy'][0]
            ax.text(0.05, 0.95, f'Total improvement: {improvement:.4f}', 
                    transform=ax.transAxes, 
                    verticalalignment='top',
                    fontsize=11,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        # Add concept names as annotations for first few interventions
        if 'intervened_concept_names' in results:
            max_annotations = min(5, len(results['num_interventions']) - 1)
            for i in range(1, max_annotations + 1):
                if i < len(results['intervened_concept_names']):
                    # Get the newly added concept name
                    new_concept = results['intervened_concept_names'][i][-1]
                    # Truncate long concept names
                    if len(new_concept) > 40:
                        new_concept = new_concept[:37] + "..."
                    
                    x = results['num_interventions'][i]
                    y = results['accuracy'][i]
                    
                    # Add annotation with arrow
                    ax.annotate(new_concept,
                               xy=(x, y), 
                               xytext=(10, -15) if i % 2 == 0 else (10, 15),
                               textcoords='offset points',
                               fontsize=9,
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7),
                               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', 
                                             color='gray', lw=1))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved as '{save_path}'")
        plt.show()
        return plt
    
    def print_intervention_summary(self, results, top_k=10):
        """Print a summary table of interventions."""
        print("\n" + "="*80)
        print("INTERVENTION SUMMARY")
        print("="*80)
        
        print(f"\nBaseline accuracy: {results['accuracy'][0]:.4f}")
        
        if len(results['accuracy']) > 1:
            print(f"\nTop {min(top_k, len(results['accuracy'])-1)} Interventions:")
            print(f"{'Rank':<6} {'Cumulative Concepts':<8} {'Accuracy':<10} {'Improvement':<12} {'New Concept'}")
            print("-" * 80)
            
            for i in range(1, min(top_k + 1, len(results['accuracy']))):
                num_int = results['num_interventions'][i]
                acc = results['accuracy'][i]
                improvement = acc - results['accuracy'][0]
                new_concept = results['intervened_concept_names'][i][-1]
                
                print(f"{i:<6} {num_int:<8} {acc:<10.4f} +{improvement:<11.4f} {new_concept}")
            
            if len(results['accuracy']) > top_k + 1:
                final_idx = len(results['accuracy']) - 1
                print("...")
                print(f"{final_idx:<6} {results['num_interventions'][final_idx]:<8} "
                      f"{results['accuracy'][final_idx]:<10.4f} "
                      f"+{results['accuracy'][final_idx] - results['accuracy'][0]:<11.4f}")
