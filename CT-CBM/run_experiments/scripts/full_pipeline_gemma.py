import sys
sys.path.append('./scripts')
sys.path.append('./run_experiments/')
sys.path.append('./run_experiments/models')
sys.path.append('./run_experiments/data')

import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
from transformers import pipeline
from concepts_discovery_utils import (
    extract_target_words, create_context_window,
    load_model, run_concepts_discovery, calculate_macro_concept_frequencies,
    update_concept_frequencies, find_most_frequent_macro_concept,
    
 ) #summarize_concepts, 
from attribution_utils import (
    process_data_in_batches, get_example, example_attribution,
    split_dataloader
)
import json

from models.utils import load_model_and_tokenizer
from models.jointCBMv2 import JointModel
from models.utils import RidgeLinearLayer


# Ranking concepts 
from ranking_utils import rank_macro_concepts, most_k_important_macro_concepts, get_concept_at_rank, randomize_scores

class JointResidualFittingModel(nn.Module):
    def __init__(self, joint_model, linear_layer, discovery_model, discovery_tokenizer, config):
        super(JointResidualFittingModel, self).__init__()

        self.joint_model = joint_model
        #the CBM module
        self.embedder_model = joint_model.embedder_model
        self.embedder_tokenizer = joint_model.embedder_tokenizer

        #the discovery module
        self.discovery_model = discovery_model
        self.discovery_tokenizer = discovery_tokenizer
        self.macro_concepts_dict = {}  # Embeddings of the macro concepts
        self.concepts_discovered_by_iteration = []

        # ranking module
        self.cavs = {}      # Initialize CAVs with an empty dictionary   
        self.tcav_scores = {}
        self.TCAVS_ranker = None

        # parameters
        self.use_cls_token = config.use_cls_token
        self.config = config
        self.model_name = config.model_name
        self.device = config.device
        self.embeddings = None
        
        # Metrics to save
        # metrics of joint model with residual layer
        self.joint_model_train_metrics = []
        self.joint_model_val_metrics = []
        self.joint_model_test_metrics = []

        # metrics of joint model with residual layer
        self.joint_model_with_r_train_metrics = []
        self.joint_model_with_r_val_metrics = []
        self.joint_model_with_r_test_metrics = []
        
        # Importance of the residual connection
        self.importance_per_iteration = []
        self.importance_per_iteration_2 = []

    # OK
    def forward(self, input_ids):
        return self.joint_model.forward_residual(input_ids)

    # OK
    def forward_pierre(self, input_ids):
        """Necessary forward for attribution calculation purpose 
        output : final_output, XtoC_output, pooled_output
        """
        return self.joint_model.forward_pierre(input_ids)

    # OK
    def get_pooled_output(self, input_ids):
        """ get the embedding of the input text"""
        return self.joint_model.get_pooled_output(input_ids)
    
    # OK
    def get_XtoY_output(self, pooled_output):
        """
        just a useful function to get the output of the XtoY layer
        """
        return self.joint_model.get_XtoY_output(pooled_output)

    # OK
    def discover_new_concepts(self, df_attribution):
        """Discover new concepts from the attribution data."""
        with torch.no_grad():
            df_attributon_2 = run_concepts_discovery(df_attribution, model=None, tokenizer=None, top_n=1, save_path=self.config.SAVE_PATH)

        return df_attributon_2
            
    # OK
    def extract_word_attributions(self, dataloader):
        """
        Extract word attributions and save them into a CSV file.
        """
        train_inputs = []
        train_masks = []
        y_train = []

        for batch in dataloader:
            for i in range(batch['input_ids'].shape[0]):
                train_inputs.append(batch['input_ids'][i])
                y_train.append(batch['label'][i])

        train_inputs = torch.stack(train_inputs).to(self.device)
        y_train = torch.stack(y_train).to(self.device)

        all_texts, all_word_attributions = process_data_in_batches(
            residual_model=self,
            tokenizer=self.embedder_tokenizer,
            inputs=train_inputs,
            masks=train_masks,
            targets=y_train,
            batch_size=1,
            example_index=0
        )

        # Create a DataFrame from the word attributions
        df_attribution = pd.DataFrame({'text': all_texts, 'word_attributions': all_word_attributions})
        
        # Save to a CSV file
        df_attribution.to_csv(f"{self.config.SAVE_PATH_CONCEPTS}/word_attribution.csv", index=False)
        return df_attribution

    # OK
    def save_model(self, path=None):
        """Save the trained model."""
        if path is None:
            os.makedirs(f"{self.config.SAVE_PATH}blue_checkpoints/{self.config.model_name}/Our_CBM_joint", exist_ok=True)
            path = f"{self.config.SAVE_PATH}blue_checkpoints/{self.config.model_name}/Our_CBM_joint/{self.model_name}_our_model.pth"
        torch.save(self.state_dict(), path)

    # OK
    def load_model(self, path=None):
        """Load the saved model."""
        if path is None:
            path = f"{self.config.SAVE_PATH}blue_checkpoints/{self.config.model_name}/Our_CBM_joint/{self.model_name}_our_model.pth"
        self.load_state_dict(torch.load(path))
        self.eval()
    

    # to modify
    def compute_connection_importance(self, train_loader):
        """
        Compute the importance of the residual connection using two different formulas:
        1. |sum(wi*xi)| / (|sum(wi*xi)| + sum(|ai*ci|))
        2. |sum(|wi*xi|)| / (|sum(|wi*xi|)| + sum(|ai*ci|))
        wi: Weights of the residual layer, xi: Inputs of the residual layer (CLS token)
        ai: Weights of the projection layer, ci: Inputs of the projection layer (concepts)
        """ 
        all_sum_wi_xi = torch.zeros(1).to(self.device)
        all_sum_ai_ci = torch.zeros(1).to(self.device)
        all_sum_wi_xi_abs = torch.zeros(1).to(self.device)
        all_sum_ai_ci_abs = torch.zeros(1).to(self.device)

        for train_batch in train_loader:
            train_inputs_ = train_batch["input_ids"].to(self.device)

            # Residual part
            with torch.no_grad():
                pooled_output = self.get_pooled_output(train_inputs_)
                wi_xi = self.joint_model.linear_layer(pooled_output)
                # print("wi_xi", wi_xi)
            all_sum_wi_xi += torch.sum(torch.sum(wi_xi, dim=1), dim=0)
            all_sum_wi_xi_abs += torch.sum(torch.sum(torch.abs(wi_xi), dim=1), dim=0)

            # CBM part
            with torch.no_grad():
                XtoY_output, XtoC = self.get_XtoY_output(pooled_output)
                # print("XtoY_output [0]", XtoY_output[0])
                ai_ci = XtoY_output[0]

            all_sum_ai_ci += torch.sum(torch.sum(ai_ci, dim=1), dim=0)
            all_sum_ai_ci_abs += torch.sum(torch.sum(torch.abs(ai_ci), dim=1), dim=0)

        # Calculate importance
        importance_original = torch.abs(all_sum_wi_xi) / (torch.abs(all_sum_wi_xi) + torch.abs(all_sum_ai_ci))
        importance_abs = torch.abs(all_sum_wi_xi_abs) / (torch.abs(all_sum_wi_xi_abs) + torch.abs(all_sum_ai_ci_abs))

        self.importance_per_iteration.append(importance_original.cpu().numpy())
        self.importance_per_iteration_2.append(importance_abs.cpu().numpy())

    def run_full_pipeline_tcavs_strategy(self, 
                                         train_loader = None, 
                                         val_loader = None, 
                                         test_loader = None, 
                                         num_iterations=3, 
                                         num_epochs_residual_layer=5, 
                                         cavs_type_arg = None, 
                                         strategy = None,
                                        coverage_threshold = None):
        """Run the full pipeline.
        Args: 
        - train_loader : DataLoader for training data augmented
        - val_loader: DataLoader for validation data
        - test_loader: DataLoader for test data
        - num_iterations: Number of iterations for learning and concept discovery
        - num_epochs_residual_layer: Number of epochs for learning the residual layer
        - cavs: cavs obtained by 'svm' or 'mean pooling or 'random'
        - save_path: Path to save the results
        """
        import time

        # Démarrer le chronométrage
        start_time = time.time()
        
        train_loader_aug = train_loader

        if strategy == 'tcavs':    
            if cavs_type_arg == 'svm' or cavs_type_arg == 'mean':
                with open(f"{self.config.SAVE_PATH}/blue_checkpoints/{self.config.model_name}/cavs/{cavs_type_arg}/sorted_macro_concepts.json", 'r') as f:
                    self.sorted_macro_concepts = json.load(f)
            else:
                print("Entrer un cavs_type_arg valide svp")
                pass
        elif strategy == 'random':
            with open(f"{self.config.SAVE_PATH}/blue_checkpoints/{self.config.model_name}/cavs/mean/sorted_macro_concepts.json", 'r') as f:
                sorted_m_c = json.load(f)
            # attribue des score de manière random au concepts
            self.sorted_macro_concepts = randomize_scores(sorted_m_c)
        elif strategy == 'frequence':
            with open(f"{self.config.SAVE_PATH_CONCEPTS}/sorted_macro_concepts_freq.json", 'r') as f:
                self.sorted_macro_concepts = json.load(f)            
        elif strategy == 'lig':
            if cavs_type_arg == 'svm' or cavs_type_arg == 'mean':
                with open(f"{self.config.SAVE_PATH}/blue_checkpoints/{self.config.model_name}/cavs/{cavs_type_arg}/sorted_macro_concepts_lig.json", 'r') as f:
                    self.sorted_macro_concepts = json.load(f)
            else:
                print("Entrer un cavs_type_arg valide svp")
                pass
        elif strategy == 'new_heuristique':
            if cavs_type_arg == 'svm' or cavs_type_arg == 'mean':
                with open(f"{self.config.SAVE_PATH}/blue_checkpoints/{self.config.model_name}/cavs/{cavs_type_arg}/sorted_macro_concepts_coverage_{self.config.annotation}_{self.config.agg_mode}_{self.config.agg_scope}.json", 'r') as f:
                    self.sorted_macro_concepts = json.load(f)
        elif strategy == 'new_heuristique_MJ':
            import pickle
            if cavs_type_arg == 'svm' or cavs_type_arg == 'mean':
                with open(f"{self.config.SAVE_PATH}/blue_checkpoints/{self.config.model_name}/cavs/{cavs_type_arg}/sorted_macro_concepts_coverage_MJ_{self.config.annotation}_{self.config.agg_mode}_{self.config.agg_scope}.pkl", 'rb') as f:
                    self.sorted_macro_concepts = pickle.load(f)
                    print(strategy)
                    print(self.sorted_macro_concepts)
        else:
            print('enter a proper strategy name among the following choice :tcavs, random, frequence, lig')

        # pour la strategy mj
        count_mj = 0
        
        # Main loop for learning and concept discovery
        for iteration in range(num_iterations):
            print(f"\n--- Iteration {iteration + 1} ---")
            
            if iteration == 0:
                # Check if concepts are present in ModelXtoCtoY_layer
                if len(self.joint_model.concepts_name) == 0:
                    
                    list_concept_first_run = []
                    found_first_coverage = False

                    if strategy != 'new_heuristique_MJ':
                        for c,v in self.sorted_macro_concepts.items():
                            if v < coverage_threshold :
                                list_concept_first_run.append(c)
                            elif v == 100 and not found_first_coverage:
                                list_concept_first_run.append(c)
                                found_first_coverage = True
                            elif found_first_coverage:
                                break 
                    else:
                        for concepts, coverage in self.sorted_macro_concepts:
                            
                            if coverage < coverage_threshold:
                                list_concept_first_run.extend(concepts)
                                count_mj += 1
                            elif coverage >= coverage_threshold and not found_first_coverage:
                                list_concept_first_run.extend(concepts)
                                found_first_coverage = True
                                count_mj += 1
                            elif found_first_coverage:
                                break

        
                    # Assurer un nombre minimal de concepts (= 3 * num_labels)
                    min_concepts = 3 * self.config.num_labels
                    if len(list_concept_first_run) < min_concepts:
                        if strategy != 'new_heuristique_MJ':
                            for c in self.sorted_macro_concepts:
                                if c not in list_concept_first_run:
                                    list_concept_first_run.append(c)
                                    if len(list_concept_first_run) >= min_concepts:
                                        break
                        else:
                            # Pour MJ, on parcourt les tuples restants
                            for concepts, _ in self.sorted_macro_concepts:
                                for c in concepts:
                                    if c not in list_concept_first_run:
                                        list_concept_first_run.append(c)
                                        if len(list_concept_first_run) >= min_concepts:
                                            break
                                if len(list_concept_first_run) >= min_concepts:
                                    break
                        print(f"[Info] Complété pour atteindre au moins {min_concepts} concepts : {list_concept_first_run}")
        
                    print(f"New concepts (itération 1): {list_concept_first_run}")
                    self.joint_model.concepts_name.extend(list_concept_first_run)
        
                    print(f"Concepts discovered so far: {self.joint_model.concepts_name}")

                    # internal renitialisation of joint_model
                    _e, _t, ModelXtoCtoY_layer, _ = load_model_and_tokenizer(self.config, n_concepts = len(list_concept_first_run))
                    CBM_joint = JointModel(self.embedder_model, self.embedder_tokenizer, ModelXtoCtoY_layer, self.config, train_loader_aug, val_loader)
                    CBM_joint.concepts_name = list_concept_first_run
    
                    # residual fitting part for pipeline below
                    linear_layer = RidgeLinearLayer(self.config.dim, self.config.num_labels, l2_lambda=self.config.l2_lambda)
                    linear_layer.to(self.device)
                    CBM_joint.linear_layer = linear_layer
    
                    # --- end of renitialisation
    
                    self.joint_model = CBM_joint
                    
                    # train for the first time the joint_model
                    self.joint_model.alternate_save = True
                    self.joint_model.strategy = strategy
                    self.joint_model.iteration = iteration
                    self.joint_model.train_model(train_loader_aug, val_loader)                    

                    train_metrics = self.joint_model.evaluate_model(train_loader_aug, 'train', metrics_on_concepts = True)
                    val_metrics = self.joint_model.evaluate_model(val_loader, 'val', metrics_on_concepts = False)
                    test_metrics = self.joint_model.evaluate_model(test_loader, 'test', metrics_on_concepts = True)

                    self.joint_model_train_metrics.append(train_metrics)
                    self.joint_model_val_metrics.append(val_metrics)
                    self.joint_model_test_metrics.append(test_metrics)
                    
                    self.concepts_discovered_by_iteration.extend(list_concept_first_run)


                    # Train the new residual layer
                    self.joint_model.train_residual_layer(train_loader_aug, num_epochs=num_epochs_residual_layer)
                    train_metrics_with_residual_layer = self.joint_model.evaluate_model_with_residual(train_loader_aug, dataset_name='Train')
                    val_metrics_with_residual_layer = self.joint_model.evaluate_model_with_residual(val_loader, dataset_name='Validation')
                    test_metrics_with_residual_layer  = self.joint_model.evaluate_model_with_residual(test_loader, dataset_name='Test')

                    self.joint_model_with_r_train_metrics.append(train_metrics_with_residual_layer)
                    self.joint_model_with_r_val_metrics.append(val_metrics_with_residual_layer)
                    self.joint_model_with_r_test_metrics.append(test_metrics_with_residual_layer)

                    self.compute_connection_importance(train_loader_aug)
                else:
                    # just collect the performance when the model if already initialize

                    train_metrics = self.joint_model.evaluate_model(train_loader_aug, 'train', metrics_on_concepts = True)
                    val_metrics = self.joint_model.evaluate_model(val_loader, 'val', metrics_on_concepts = False)
                    test_metrics = self.joint_model.evaluate_model(test_loader, 'test', metrics_on_concepts = True)

                    self.joint_model_train_metrics.append(train_metrics)
                    self.joint_model_val_metrics.append(val_metrics)
                    self.joint_model_test_metrics.append(test_metrics)

                    self.concepts_discovered_by_iteration.append(self.joint_model.concepts_name)

                    # Train the new residual layer
                    self.joint_model.iteration = iteration
                    self.joint_model.train_residual_layer(train_loader_aug, num_epochs=num_epochs_residual_layer)
                    train_metrics_with_residual_layer = self.joint_model.evaluate_model_with_residual(train_loader_aug, dataset_name='Train')
                    val_metrics_with_residual_layer = self.joint_model.evaluate_model_with_residual(val_loader, dataset_name='Validation')
                    test_metrics_with_residual_layer  = self.joint_model.evaluate_model_with_residual(test_loader, dataset_name='Test')

                    self.joint_model_with_r_train_metrics.append(train_metrics_with_residual_layer)
                    self.joint_model_with_r_val_metrics.append(val_metrics_with_residual_layer)
                    self.joint_model_with_r_test_metrics.append(test_metrics_with_residual_layer)

                    self.compute_connection_importance(train_loader_aug)
            else:            

                # START BY COMPARING PERF WITH AND WITHOUT RESIDUAL CONNECTION

                # REMINDER: evaluate_model_with_residual RETURN val_loss, val_accuracy, val_macro_f1
                if self.joint_model_with_r_val_metrics[iteration-1][1] <= self.joint_model_val_metrics[iteration-1]['task_global_accuracy']:
                    print("criteria on val set : stop at iteratinon", iteration - 1)
                    print("perfomance with residual (ON VAL): accuracy at", self.joint_model_with_r_val_metrics[iteration-1][1])
                    print("perfomance without residual (ON VAL): accuracy at", self.joint_model_val_metrics[iteration-1]['task_global_accuracy'])

                    print("perfomance with residual (ON TEST) : accuracy at", self.joint_model_with_r_test_metrics[iteration-1][1])
                    print("perfomance without residual (ON TEST) : accuracy at", self.joint_model_test_metrics[iteration-1]['task_global_accuracy'])

                    with open(f"{self.config.SAVE_PATH}blue_checkpoints/{self.config.model_name}/Our_CBM_joint/{self.model_name}_performance_{strategy}_MJ.json", 'w') as f:
                        json.dump({
                            "concepts_discovered_by_iteration": self.concepts_discovered_by_iteration,
                            "train_metrics": self.joint_model_train_metrics,
                            "val_metrics": self.joint_model_val_metrics,
                            "test_metrics": self.joint_model_test_metrics,
                            "train_metrics_with_residual_layer": self.joint_model_with_r_train_metrics,
                            "val_metrics_with_residual_layer": self.joint_model_with_r_val_metrics,
                            "test_metrics_with_residual_layer": self.joint_model_with_r_test_metrics,
                            "importance_per_iteration_model": self.importance_per_iteration,
                            "importance_per_iteration_model_2": self.importance_per_iteration_2,
                        }, f, default=lambda x: x.tolist())
                        
                    break
                else:
                    # CHOOSE THE SECOND BATCH OF CONCEPT TO ADD
                    if strategy != 'new_heuristique_MJ':
                        nb_new_concepts = 1  # Par exemple, à adapter selon tes besoins
                        new_concepts = []
                
                        for c, _ in self.sorted_macro_concepts.items():
                            if c not in self.joint_model.concepts_name:
                                new_concepts.append(c)
                                if len(new_concepts) == nb_new_concepts:
                                    break
                        print(f"New selected concepts for iteration {iteration + 1}: {new_concepts}")              
                                           
                        self.joint_model.concepts_name.extend(new_concepts)
                    else:
                        # 1 1 1 1 Strategy 
                        nb_new_concepts = 1  # Par exemple, à adapter selon tes besoins
                        new_concepts = []
            
                        path_combined_score_LIG = (f"{self.config.SAVE_PATH}/blue_checkpoints/{self.config.model_name}/cavs"
                        f"/{self.config.cavs_type}/combined_score_LIG_{self.config.annotation}_{self.config.agg_mode}_{self.config.agg_scope}.json"
                        )
                        with open(path_combined_score_LIG, 'r') as f:
                            self.sorted_macro_concepts = json.load(f)

                        for c, _ in self.sorted_macro_concepts.items():
                            if c not in self.joint_model.concepts_name:
                                new_concepts.append(c)
                                if len(new_concepts) == nb_new_concepts:
                                    break
                        print(f"New selected concepts for iteration {iteration + 1}: {new_concepts}")              
                                           
                        self.joint_model.concepts_name.extend(new_concepts)       
                    
                    concepts_name_iter = self.joint_model.concepts_name
    
                    # internal renitialisation of joint_model
                    _e, _t, ModelXtoCtoY_layer, _ = load_model_and_tokenizer(self.config, n_concepts = len(concepts_name_iter))
                    CBM_joint = JointModel(self.embedder_model, self.embedder_tokenizer, ModelXtoCtoY_layer, self.config, train_loader_aug, val_loader)
                    CBM_joint.concepts_name = concepts_name_iter
    
                    # residual fitting part for pipeline below
                    linear_layer = RidgeLinearLayer(self.config.dim, self.config.num_labels, l2_lambda=self.config.l2_lambda)
                    linear_layer.to(self.device)
                    CBM_joint.linear_layer = linear_layer
    
                    # --- end of renitialisation
    
                    self.joint_model = CBM_joint
                    self.joint_model.alternate_save = True
                    self.joint_model.strategy = strategy
                    self.joint_model.iteration = iteration
                    self.joint_model.train_model(train_loader_aug, val_loader)
    
                    train_metrics = self.joint_model.evaluate_model(train_loader_aug, 'train', metrics_on_concepts = True)
                    val_metrics = self.joint_model.evaluate_model(val_loader, 'val', metrics_on_concepts = False)
                    test_metrics = self.joint_model.evaluate_model(test_loader, 'test', metrics_on_concepts = True)
    
                    self.joint_model_train_metrics.append(train_metrics)
                    self.joint_model_val_metrics.append(val_metrics)
                    self.joint_model_test_metrics.append(test_metrics)
    
                    self.concepts_discovered_by_iteration.extend(new_concepts)
    
                    # Train the new residual layer
                    self.joint_model.train_residual_layer(train_loader_aug, num_epochs=num_epochs_residual_layer)
                    train_metrics_with_residual_layer = self.joint_model.evaluate_model_with_residual(train_loader_aug)
                    val_metrics_with_residual_layer = self.joint_model.evaluate_model_with_residual(val_loader)
                    test_metrics_with_residual_layer  = self.joint_model.evaluate_model_with_residual(test_loader)
    
                    self.joint_model_with_r_train_metrics.append(train_metrics_with_residual_layer)
                    self.joint_model_with_r_val_metrics.append(val_metrics_with_residual_layer)
                    self.joint_model_with_r_test_metrics.append(test_metrics_with_residual_layer)
             
                    self.compute_connection_importance(train_loader_aug)
            
                with open(f"{self.config.SAVE_PATH}blue_checkpoints/{self.config.model_name}/Our_CBM_joint/{self.model_name}_performance_{strategy}.json", 'w') as f:
                    json.dump({
                        "concepts_discovered_by_iteration": self.concepts_discovered_by_iteration,
                        "train_metrics": self.joint_model_train_metrics,
                        "val_metrics": self.joint_model_val_metrics,
                        "test_metrics": self.joint_model_test_metrics,
                        "train_metrics_with_residual_layer": self.joint_model_with_r_train_metrics,
                        "val_metrics_with_residual_layer": self.joint_model_with_r_val_metrics,
                        "test_metrics_with_residual_layer": self.joint_model_with_r_test_metrics,
                        "importance_per_iteration_model": self.importance_per_iteration,
                        "importance_per_iteration_model_2": self.importance_per_iteration_2,
                    }, f, default=lambda x: x.tolist())

        print("Finished the full pipeline execution.")
        # Arrêter le chronométrage
        end_time = time.time()
    
        # Calculer le temps total écoulé
        elapsed_time = end_time - start_time
        print(f"Temps total d'exécution : {elapsed_time:.2f} secondes")
        
        self.save_model()
