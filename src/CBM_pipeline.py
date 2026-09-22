import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
import torch.nn.functional as F
from torchvision import transforms, datasets
from transformers import BlipModel, BlipProcessor
from tqdm import tqdm
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold
import numpy as np
import glob
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score, classification_report
import re
from functools import lru_cache
import multiprocessing
import mmap
import json
import pickle

from transformers import BlipModel, BlipProcessor, CLIPModel, CLIPProcessor

from baseline_model import CustomClassifier, plot_history
from CBM_model import CBM_model, plot_concept_to_class_weights, plot_interconcept_leakage_heatmaps, plot_leakage_graph_for_class, plot_interconcept_leakage_by_cluster, plot_leakage_graph_by_activation, plot_leakage_graph_by_cosine_similarity, plot_leakage_score_by_co_activation, heatmap_leakage_score_by_co_activation,plot_leakage_graph_by_concept_weight_similarity, plot_leakage_graph_by_semantic_similarity,plot_leakage_graph_by_concept_co_activation, plot_tast_leakage_by_activation_similarity,plot_leakage_bar_graph_by_activation,plot_task_leakage_by_weight_similarity, plot_task_vs_interconcept_leakage
from data_N24_concepts import PATH_DATA_N24, load_data_N24
from data_CUB_multimodal import PATH_DATA_CUB, load_data_CUB
from data_agnews_concepts import load_data_agnews, PATH_DATA_agnews
from data_dbpedia_concepts import load_data_dbpedia, PATH_DATA_dbpedia
from KAN import plot_concept_to_class_response_curves

import matplotlib.pyplot as plt
from adjustText import adjust_text
from matplotlib.patches import Patch
from matplotlib.cm import get_cmap

# PATH_FIGURES = "/data/graphs/N24_News"
PATH_FIGURES = "/data/graphs/N24_News"


###############################################################################
# parameters 
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_workers = multiprocessing.cpu_count()
max_len = 512
cluster_json_path = os.path.join(PATH_DATA_N24, "augmented_dataset_C3M/cluster_dict_strat_reassignation_C3M.json")

###############################################################################
def run_baseline(dataset='N24', backbone='clip', num_epochs=10, load=False, contrastive=False, modality='multi', on_concept=False, combine_type='concat', import_concept_list='', concept_level=0, select_concepts=None, frozen_backbone=False):

    """
    Train and evaluate a backbone model (X to Y, no concept layer)
    If on_concept = True, predict concept labels (X to C)
    
    Parameters
    ----------
        dataset : str, default='N24'
            Dataset to use. Multimodal options: 'N24', 'CUB'. Text-only options: 'agnews', 'dbpedia'.
        backbone : str, default='clip'
            Backbone model to use. Options: 'clip' or 'clip-large'.
        num_epochs : int, default=10
            Number of training epochs.
        load : bool, default=False
            Whether to load a pre-trained CBM.
        contrastive : bool, default=False
            If True, use contrastive learning for the backbone.
        modality : str, default='multi'
            Modality to use. Options: 'multi' (image and text), 'image', 'text'.
        on_concept : bool, default=False
            If True, train the model on concepts instead of final classes (for tests only).
        combine_type : str, default='combine'
            How to combine concept modalities. Options: 'combine' (combined concept modalities) or 'concat' (separate multimodal concept predictions).
        import_concept_list : str, default=''
            Path to a file containing a list of concepts to use (from heuristic).
        concept_level : int, default=0
            Which level to select from the heuristic level file.
        select_concepts : list, default=None
            List of concepts to select from the dataset
        frozen_backbone : bool, default=False
            If True, freeze the backbone during training.
    
    Returns
    -------
        history : dict
            Dictionary containing all training and test information for plots.
        model : object
            The best trained model.
    """


    if(import_concept_list!=''):
        with open(import_concept_list, 'rb') as f:
            concept_ranking = pickle.load(f)
        # concat lists from concept level = 0 until concept_level value
        select_concepts = []
        for level in range(concept_level+1):
            select_concepts += concept_ranking[level][0]
        print('import concept list')
        print(select_concepts)

    if(dataset=='N24'):
        train_loader, val_loader, test_loader, class_dict, concept_list, concept_counts = load_data_N24(data_dir=PATH_DATA_N24, class_list=None, batch_size=batch_size, max_len=max_len)
        path_export = PATH_DATA_N24
    elif(dataset=='CUB'):
        train_loader, val_loader, test_loader, class_dict, concept_list, concept_counts = load_data_CUB(data_dir=PATH_DATA_CUB, batch_size=batch_size)
        path_export = PATH_DATA_CUB
    elif(dataset=='agnews'):
        train_loader, val_loader, test_loader, class_dict, concept_list, concept_counts = load_data_agnews(batch_size=batch_size)
        path_export = PATH_DATA_agnews
    elif(dataset=='dbpedia'):
        train_loader, val_loader, test_loader, class_dict, concept_list, concept_counts = load_data_dbpedia(batch_size=batch_size)
        path_export = PATH_DATA_dbpedia
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    # import backbone
    if(backbone=='clip'):
        base_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    elif(backbone=='clip-large'):
        base_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    elif(backbone=='blip'):
        base_model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

    model = CustomClassifier(device=device, class_dict=class_dict, base_model=base_model, processor=processor, num_classes=len(class_dict), classifier_type='simple', frozen_backbone=frozen_backbone, modality=modality)

    if(not load):

        if(not contrastive):
            history, best_model = model.train_model(train_dataloader=train_loader, val_dataloader=val_loader, test_dataloader=test_loader, max_len=max_len, num_epochs=num_epochs)
        else:
            history, best_model = model.contrastive_train(train_dataloader=train_loader, val_dataloader=val_loader, test_dataloader=test_loader, num_epochs=num_epochs, on_concept=on_concept, concept_list=concept_list)

        # check if save path exists, if not create folder
        if not os.path.exists(f"{path_export}/models"):
            os.makedirs(f"{path_export}/models")

        torch.save({'model_state_dict': best_model},f"{path_export}/models/{backbone}_baseline.pt")

        plot_history(history, description='training of CLIP with simple classifier', training_mode=True)

    else:
        print("loading model")
        model.load_model(f"{path_export}/models/{backbone}_baseline.pt")

        if(not contrastive):
            print("evaluating model")
            test_loss, test_accuracy, test_f1, test_class_f1 = model.evaluate(test_loader)
        else:
            test_loss, test_accuracy, test_f1, test_class_f1 = model.contrastive_evaluate(test_loader, on_concept=on_concept, concept_list=concept_list)

        history['test_acc'] = test_accuracy
        history['test_f1'] = test_f1

        # plot_history(history, description='training of CLIP with simple classifier', training_mode=False)

    return history, model

###############################################################################
def run_CBM(dataset='N24',dataset_type='C3M', combine_type='combine', backbone='clip', concept_representation='logits', num_epochs=10, select_most_frequent=None, select_least_frequent=None, select_concepts=None, random_concepts=None, load=False, sequential=False, independant=False, relu_concepts=False, lambda_XtoC=1, lambda_XtoC_leakage=1, cpo=False, learning_rate=1e-5, fixed_lr=False, plot = False, import_concept_list='', concept_level=0, contrastive=False, frozen_backbone=False, loss_rescaled=True, leakage_loss=False, leakage_loss_activation='up', loss_CBLLM='MSE', save_model=False, kan_layer=False):
    """
    Train and evaluate a Concept Bottleneck Model (CBM).
    
    Parameters
    ----------
        dataset : str, default='N24'
            Dataset to use. Multimodal options: 'N24', 'CUB'. Text-only options: 'agnews', 'dbpedia'.
        dataset_type : str, default='C3M'
            Type of dataset. Options: 'C3M' or 'CBLLM'.
        combine_type : str, default='combine'
            How to combine concept modalities. Options: 'combine' (combined concept modalities) or 'concat' (separate multimodal concept predictions). Or possibility for unimodal training by using 'text' or 'image'.
        backbone : str, default='clip'
            Backbone model to use. Options: 'clip' or 'clip-large'.
        concept_representation : str, default='logits'
            Activation function for concept representations. For C3M: 'logits', 'sigmoid', or 'gumbel-sigmoid'. For CBLLM: 'importance' (= raw logits).
        num_epochs : int, default=10
            Number of training epochs.
        select_most_frequent : int or None, default=None
            If specified, select this many concepts with highest activations. Works only for N24 dataset.
        select_least_frequent : int or None, default=None
            If specified, select this many concepts with lowest activations. Works only for N24 dataset.
        select_concepts : list of str or None, default=None
            List of specific concept names to include in training.
        random_concepts : int or None, default=None
            If specified, randomly select this many concepts from the dataset.
        load : bool, default=False
            Whether to load a pre-trained CBM.
        sequential : bool, default=False
            If True, train CBM sequentially.
        independant : bool, default=False
            If True, train CBM independently.
        relu_concepts : bool, default=False
            If True, apply ReLU activation to concept activations. Only valid when concept_representation is 'logits' or 'importance'.
        lambda_XtoC : float, default=1
            Weight of the concept loss relative to the final task loss.
        lambda_XtoC_leakage : float, default=1
            Weight of the concept leakage loss relative to the final task loss.
        cpo : bool, default=False
            If True, train concept layer with Concept Preference Optimization loss (arXiv:2504.18026).
        learning_rate : float, default=1e-5
            Learning rate for the backbone.
        fixed_lr : bool, default=False
            If True, use fixed learning_rate for all model parts. If False, apply cosine annealing for linear layers (starting at 1e-1, ending at 1e-5).
        plot : bool, default=False
            If True, plot the training history.
        import_concept_list : str, default=''
            Path to a file containing a list of concepts to use (from heuristic).
        concept_level : int, default=0
            Which level to select from the heuristic level file.
        contrastive : bool, default=False
            If True, train CBM in contrastive mode.
        frozen_backbone : bool, default=False
            If True, freeze the backbone during training.
        loss_rescaled : bool, default=True
            If True, rescale concept loss to match the average order of magnitude of the final task loss. Also applies to leakage loss when activated.
        leakage_loss : bool, default=False
            If True, add a leakage loss to the concept loss.
        leakage_loss_activation : str, default='up'
            Leakage loss weight schedule. Options: 'up' (cosine annealing increasing), 'down' (cosine annealing decreasing), or any other value (take leakage loss as it is).
        loss_CBLLM : str, default='MSE'
            Loss function for CBLLM. Options: 'MSE' or 'cos_cubed' (from label-free article arXiv:2304.06129).
        save_model : bool or str, default=False
            If True, save the trained model. If a string, append this string to the output model name. Required for importing models with appended names.
        kan_layer : bool, default=False
            If True, replace the linear final layer (C to Y) with a KAN layer (arXiv:2404.19756v5).
    
    Returns
    -------
        history : dict
            Dictionary containing all training and test information for plots.
        model : object
            The best trained model.
    """

    if(import_concept_list!=''):

        _, ext = os.path.splitext(import_concept_list)  # ext like ".pkl" or ".json"

        if ext.lower() in ('.pkl', '.pickle'):
            with open(import_concept_list, 'rb') as f:
                concept_ranking = pickle.load(f)

                # concat lists from concept level = 0 until concept_level value
                select_concepts = []
                for level in range(concept_level+1):
                    select_concepts += concept_ranking[level][0]
                    coverage = concept_ranking[level][1]
                print('import concept list')
                print('coverage is', coverage, ' valid = ', coverage > 99)

        elif ext.lower() == '.json':
            with open(import_concept_list, 'r') as f:
                concept_ranking = json.load(f)

            # select first concepts until number concept_level
            select_concepts = list(concept_ranking.keys())[:concept_level]
            print('import concept list')

        else:
            raise ValueError(f"Unsupported concept list format: {ext}")


    # train_loader, val_loader, test_loader, class_dict, concept_list, concept_counts, info_dict = load_data_N24(data_dir=PATH_DATA_N24, class_list=None, batch_size=batch_size, max_len=max_len, dataset_type=dataset_type, combine_type=combine_type, select_most_frequent=select_most_frequent, select_least_frequent=None, select_concepts=select_concepts, random_concepts=random_concepts)
    if(dataset=='N24'):
        train_loader, val_loader, test_loader, class_dict, concept_list, concept_counts = load_data_N24(data_dir=PATH_DATA_N24, class_list=None, batch_size=batch_size, max_len=max_len, dataset_type=dataset_type, combine_type=combine_type, select_most_frequent=select_most_frequent, select_least_frequent=select_least_frequent, select_concepts=select_concepts, random_concepts=random_concepts)
        path_export = PATH_DATA_N24
    elif(dataset=='CUB'):
        train_loader, val_loader, test_loader, class_dict, concept_list, concept_counts = load_data_CUB(data_dir=PATH_DATA_CUB, batch_size=batch_size, dataset_type=dataset_type, combine_type=combine_type, select_concepts=select_concepts, random_concepts=random_concepts)
        path_export = PATH_DATA_CUB
    elif(dataset=='agnews'):
        train_loader, val_loader, test_loader, class_dict, concept_list, concept_counts = load_data_agnews(batch_size=batch_size, dataset_type=dataset_type, combine_type=combine_type, select_concepts=select_concepts, random_concepts=random_concepts)
        path_export = PATH_DATA_agnews
    elif(dataset=='dbpedia'):
        train_loader, val_loader, test_loader, class_dict, concept_list, concept_counts = load_data_dbpedia(batch_size=batch_size, dataset_type=dataset_type, combine_type=combine_type, select_concepts=select_concepts, random_concepts=random_concepts)
        path_export = PATH_DATA_dbpedia
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    # display number of concepts
    print(f"Number of concepts: {len(concept_list)}", ' valid (> 3*n_classes) = ', len(concept_list) > 3*len(class_dict))

    print(concept_list)

    # import backbone
    if(backbone=='clip'):
        base_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    elif(backbone=='clip-large'):
        base_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    elif(backbone=='blip'):
        base_model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

    if(dataset_type=='CBLLM'):
        concept_representation = 'importance'

    # create CBM
    # model = CBM_model(device=device, class_dict=class_dict, base_model=base_model, processor=processor, num_classes=len(class_dict), concept_list=concept_list, concept_training=True, frozen_backbone=False, concept_representation=concept_representation, lambda_XtoC=lambda_XtoC, relu_concepts=relu_concepts, cpo=cpo, contrastive=contrastive)
    model = CBM_model(device=device, class_dict=class_dict, base_model=base_model, processor=processor, num_classes=len(class_dict), concept_list=concept_list, concept_training=True, frozen_backbone=frozen_backbone, concept_representation=concept_representation, lambda_XtoC=lambda_XtoC, lambda_XtoC_leakage=lambda_XtoC_leakage, relu_concepts=relu_concepts, cpo=cpo, loss_rescaled=loss_rescaled, leakage_loss=leakage_loss, leakage_loss_activation=leakage_loss_activation, loss_CBLLM=loss_CBLLM, kan_layer=kan_layer)

    sequential_info = ''
    if(sequential):
        sequential_info = '_sequential'
    independant_info = ''
    if(independant):
        independant_info = '_independant'
    relu_info = ''
    if(relu_concepts):
        relu_info = '_relu'
    cpo_info = ''
    if(cpo):
        cpo_info = '_cpo'

    if(not load):

        if(sequential):
            history, best_model = model.sequential_train_model(train_dataloader=train_loader, val_dataloader=val_loader, test_dataloader=test_loader, max_len=max_len, num_epochs=num_epochs, learning_rate=learning_rate, fixed_lr=fixed_lr)
        
        elif(independant):
            history, best_model = model.independant_train_model(train_dataloader=train_loader, val_dataloader=val_loader, test_dataloader=test_loader, max_len=max_len, num_epochs=num_epochs, learning_rate=learning_rate, fixed_lr=fixed_lr)

        else:
            history, best_model = model.train_model(train_dataloader=train_loader, val_dataloader=val_loader, test_dataloader=test_loader, max_len=max_len, num_epochs=num_epochs, learning_rate=learning_rate, fixed_lr=fixed_lr)

        history['concept_counts'] = concept_counts
        # history.update(info_dict)

        print(history)

        if(save_model):

            # check if save path exists, if not create folder
            if not os.path.exists(f"{path_export}/models"):
                os.makedirs(f"{path_export}/models")

            # append save_model to output name if save_model is a string
            if(type(save_model) == str):
                torch.save({'model_state_dict': best_model}, f"{path_export}/models/{backbone}_CBM_{concept_representation}_{combine_type}_{int(num_epochs)}epochs{sequential_info}{independant_info}_{len(concept_list)}concepts{relu_info}{cpo_info}_lambdaXtoC_{lambda_XtoC}_{save_model}.pt") # save best model
                print("saving model at :")
                print(f"{path_export}/models/{backbone}_CBM_{concept_representation}_{combine_type}_{int(num_epochs)}epochs{sequential_info}{independant_info}_{len(concept_list)}concepts{relu_info}{cpo_info}_lambdaXtoC_{lambda_XtoC}_{save_model}.pt")
            else:
                torch.save({'model_state_dict': best_model}, f"{path_export}/models/{backbone}_CBM_{concept_representation}_{combine_type}_{int(num_epochs)}epochs{sequential_info}{independant_info}_{len(concept_list)}concepts{relu_info}{cpo_info}_lambdaXtoC_{lambda_XtoC}.pt") # save best model
                print("saving model at :")
                print(f"{path_export}/models/{backbone}_CBM_{concept_representation}_{combine_type}_{int(num_epochs)}epochs{sequential_info}{independant_info}_{len(concept_list)}concepts{relu_info}{cpo_info}_lambdaXtoC_{lambda_XtoC}.pt")

            
        
        if plot :
            if(dataset_type=='CBLLM'):
                plot_history(history, description=f'training of {backbone} CBM ({concept_representation} {combine_type}) with simple classifier (with concept training)', cbllm=True)
            else:
                plot_history(history, description=f'training of {backbone} CBM ({concept_representation} {combine_type}) with simple classifier (with concept training)')

            if((model.combine_type!='text') and (model.combine_type!='image')):
                modality_scores = model.compute_modality_score(test_loader)
                plot_modality_scores(modality_scores, concept_list)

            if(not kan_layer):
                plot_concept_to_class_weights(model, num_top_concepts=5)
            else:
                plot_concept_to_class_response_curves(model.classifier, train_loader, concept_list, class_dict)

    else:
        if(type(save_model) == str):
            model.load_model(f"{path_export}/models/{backbone}_CBM_{concept_representation}_{combine_type}_{int(num_epochs)}epochs{sequential_info}{independant_info}_{len(concept_list)}concepts{relu_info}{cpo_info}_lambdaXtoC_{lambda_XtoC}_{save_model}s.pt")
        else:
            model.load_model(f"{path_export}/models/{backbone}_CBM_{concept_representation}_{combine_type}_{int(num_epochs)}epochs{sequential_info}{independant_info}_{len(concept_list)}concepts{relu_info}{cpo_info}_lambdaXtoC_{lambda_XtoC}.pt")
        # model.load_model(f"{path_export}/models/{backbone}_CBM_{concept_representation}_{combine_type}_{int(num_epochs)}epochs{sequential_info}{independant_info}_{len(concept_list)}concepts{relu_info}{cpo_info}.pt")
        print("model loaded")
        concept_preds, avg_loss, accuracy, f1, concept_acc, per_concept_acc, concept_f1, per_concept_acc, per_concept_f1, concept_leakage, per_concept_leakage, interconcept_leakage = model.evaluate(test_loader)
        # add metrics to history
        history = {}

        history['concept_counts'] = concept_counts
        # history.update(info_dict)

        history['test_acc'] = accuracy
        history['test_f1'] = f1
        print(f"test accuracy: {accuracy}")
        print(f"test f1: {f1}")
        
        history['test_concept_acc'] = concept_acc
        history['test_per_concept_acc'] = per_concept_acc
        history['test_concept_f1'] = concept_f1
        history['test_per_concept_f1'] = per_concept_f1
        history['test_concept_leakage'] = concept_leakage
        history['test_per_concept_leakage'] = per_concept_leakage
        history['interconcept_leakage'] = interconcept_leakage

        if plot :
            plot_interconcept_leakage_heatmaps(interconcept_leakage, concept_list)
            # for class_idx in range(model.num_classes):
            #     plot_leakage_graph_for_class(model, interconcept_leakage, class_idx, num_top_concepts=10)

            # plot_interconcept_leakage_by_cluster(
            #     interconcept_leakage,
            #     concept_list,
            #     cluster_json_path
            # )

            # plot_leakage_graph_by_activation(concept_preds, interconcept_leakage, concept_list, per_concept_acc, per_concept_leakage, num_top_concepts=10)
            # # plot_leakage_graph_by_cosine_similarity(concept_list, interconcept_leakage, model)
            plot_leakage_graph_by_concept_weight_similarity(concept_list, interconcept_leakage, model, save_to_dbfs=True, path = PATH_FIGURES)
            # plot_leakage_graph_by_semantic_similarity(concept_list, interconcept_leakage)
            plot_leakage_graph_by_concept_co_activation(concept_preds, concept_list, interconcept_leakage, model)
            plot_tast_leakage_by_activation_similarity(concept_preds, concept_list, per_concept_acc, per_concept_leakage)
            plot_leakage_bar_graph_by_activation(concept_preds, interconcept_leakage, concept_list, per_concept_acc, per_concept_leakage, save_to_dbfs=True, path = PATH_FIGURES)
            plot_task_leakage_by_weight_similarity(model,concept_list,per_concept_acc, per_concept_leakage)
            plot_task_vs_interconcept_leakage(interconcept_leakage,per_concept_leakage,concept_list,per_concept_acc=None, save_to_dbfs=True, path = PATH_FIGURES)
            # plot_leakage_score_by_co_activation(concept_preds, interconcept_leakage, concept_list)
            # heatmap_leakage_score_by_co_activation(concept_preds, interconcept_leakage, concept_list)

            # plot_history(history, description=f'training of {backbone} CBM ({concept_representation} {combine_type}) with simple classifier (with concept training)', training_mode=False)

            if((model.combine_type!='text') and (model.combine_type!='image')):
                modality_scores = model.compute_modality_score(test_loader)
                plot_modality_scores(modality_scores, concept_list)

            if(not kan_layer):
                plot_concept_to_class_weights(model, num_top_concepts=5)
        

    # print(history)

    return history, model

###############################################################################
def experiments(dataset_type='C3M', backbone='clip', num_epochs=10, select_most_frequent=None, load=False, cpo=False):

    #combine_types=['combine', 'concat']
    combine_types=['combine']
    concept_representations=['logits','sigmoid','gumbel_sigmoid'] # 'relu'
    trainings=['joint','sequential','independant']
    #cpos=[False]

    # for tests
    #combine_types=['combine']
    #concept_representations=['logits']
    #sequentials=[False]

    all_history = {}

    # run baseline
    history = run_baseline(backbone=backbone, num_epochs=num_epochs)
    all_history['baseline'] = history
    #all_history['baseline'] = {'test_acc': 0.9703333333333334, 'test_f1': 0.9703029165841186, 'test_concept_acc': 0., 'test_concept_f1': 0.}

    # run_CBM with different parameters
    for combine_type in combine_types:
        for concept_representation in concept_representations:
            for training in trainings:
                #for cpo in cpos:
                print('----------------------------------------------------------------------------')
                print(f'Running {backbone} CBM with concept rep = {concept_representation}, concept dataset = {combine_type}, training =  {training}, cpo = {cpo}')
                print('----------------------------------------------------------------------------')

                sequential = False
                independant = False
                if(training=='sequential'):
                    sequential = True
                elif(training=='independant'):
                    independant = True

                relu_concepts = False
                concept_representation2 = concept_representation
                if(concept_representation=='relu'):
                    relu_concepts = True
                    concept_representation2 = 'logits'

                cpo_label = ''
                if(cpo):
                    cpo_label = '_cpo'

                history, _ = run_CBM(dataset_type=dataset_type, combine_type=combine_type, backbone=backbone, concept_representation=concept_representation2, num_epochs=num_epochs, select_most_frequent=select_most_frequent, relu_concepts=relu_concepts, load=load, sequential=sequential, independant=independant, cpo=cpo)
                # record history for these parameters
                all_history[f'{concept_representation}_{combine_type}_{training}{cpo_label}'] = history

    print(all_history)

    plot_experiment_results(all_history, info=f'CBM with {backbone} and {dataset_type} dataset')

    plot_leakage_results(all_history, info=f'CBM with {backbone} and {dataset_type} dataset')

    return all_history


###############################################################################
def plot_experiment_results(all_history, info=''):
    # Liste de marqueurs différents (tu peux en ajouter d'autres si besoin)
    marker_list = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', '<', '>', 'h', 'H', 'p', '+', 'x', 'd', '|', '_']
    
    # Extract metrics for all experiments
    x_acc, y_acc, x_f1, y_f1, labels = [], [], [], [], []

    for key, results in all_history.items():
        y_acc.append(results['test_acc'])
        y_f1.append(results['test_f1'])
        if key != 'baseline':
            x_acc.append(results['test_concept_acc'])
            x_f1.append(results['test_concept_f1'])
        else:
            x_acc.append(0)
            x_f1.append(0)
        labels.append(key)

    # Assign a unique color to each label
    cmap = get_cmap('tab10')
    colors = [cmap(i % 10) for i in range(len(labels))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12))

    # Plot 1: Accuracy
    for i, (x, y) in enumerate(zip(x_acc, y_acc)):
        marker = marker_list[i % len(marker_list)]
        ax1.scatter(x, y, color=colors[i], s=300, alpha=0.5, marker=marker)
    texts = []
    for i, label in enumerate(labels):
        ann = ax1.annotate(label, (x_acc[i], y_acc[i]), fontsize=10, color=colors[i])
        texts.append(ann)
        ax1.plot([x_acc[i], ann.get_position()[0]],
                 [y_acc[i], ann.get_position()[1]],
                 color=colors[i], linestyle='--', linewidth=1)
    adjust_text(
    texts,
    ax=ax1,
    arrowprops=dict(arrowstyle='-', color='gray'),
    force_points=10,
    force_text=10,
    expand_points=(4, 4),
    expand_text=(4, 4),
    expand=(3, 3),  # Ajoute aussi ce paramètre pour élargir la zone de répulsion globale
)
    ax1.set_xlabel('Test Concept Accuracy', fontsize=20)
    ax1.set_ylabel('Test Accuracy', fontsize=20)
    ax1.set_title(info, fontsize=24)
    ax1.tick_params(axis='both', which='major', labelsize=18)

    # Plot 2: F1 Score
    for i, (x, y) in enumerate(zip(x_f1, y_f1)):
        marker = marker_list[i % len(marker_list)]
        ax2.scatter(x, y, color=colors[i], s=300, alpha=0.5, marker=marker)
    texts = []
    for i, label in enumerate(labels):
        ann = ax2.annotate(label, (x_f1[i], y_f1[i]), fontsize=10, color=colors[i])
        texts.append(ann)
        ax2.plot([x_f1[i], ann.get_position()[0]],
                 [y_f1[i], ann.get_position()[1]],
                 color=colors[i], linestyle='--', linewidth=1)
    adjust_text(
    texts,
    ax=ax2,
    arrowprops=dict(arrowstyle='-', color='gray'),
    force_points=10,
    force_text=10,
    expand_points=(4, 4),
    expand_text=(4, 4),
    expand=(3, 3),  # Ajoute aussi ce paramètre pour élargir la zone de répulsion globale
)
    ax2.set_xlabel('Test Concept F1', fontsize=20)
    ax2.set_ylabel('Test F1', fontsize=20)
    ax2.set_title(info, fontsize=24)
    ax2.tick_params(axis='both', which='major', labelsize=18)

    # plot limits
    acc_x_min = min(x_acc)
    acc_y_min = min(y_acc)
    f1_x_min = min(x_f1)
    f1_y_min = min(y_f1)

    ax1.set_xlim(min(acc_x_min,f1_x_min), 1)
    ax1.set_ylim(0.93+0*min(acc_y_min, f1_y_min), 1)    
    ax2.set_xlim(min(acc_x_min,f1_x_min), 1)
    ax2.set_ylim(0.93+0*min(acc_y_min, f1_y_min), 1)

    # Create a single legend for both plots, sur plusieurs lignes et bien au-dessus
    legend_handles = [Patch(color=colors[i], label=labels[i]) for i in range(len(labels))]
    ncol = 4 if len(labels) > 4 else len(labels)
    fig.legend(handles=legend_handles, loc='lower center', ncol=ncol, fontsize=16, bbox_to_anchor=(0.5, 0.9))
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()

###############################################################################
def plot_leakage_results(all_history, info=''):
    """
    Plots a bar chart of test_concept_leakage scores for different training strategies.

    Parameters:
        all_history (dict): Keys are training strategies, values are dicts with key 'test_concept_leakage'.
        info (str): Title info for the plot.
    """
    # Extract strategies and their leakage scores
    strategies = [strategy for strategy in list(all_history.keys()) if 'test_concept_leakage' in all_history[strategy].keys()]
    leakage_scores = [all_history[strategy]['test_concept_leakage'] for strategy in strategies if 'test_concept_leakage' in all_history[strategy].keys()]

    # Create bar plot
    plt.figure(figsize=(8, 5))
    bars = plt.bar(strategies, leakage_scores, color='skyblue')
    plt.xlabel('Training Strategy')
    plt.ylabel('Test Concept Leakage')
    plt.title(info)
    plt.ylim(0, 1)

    # Annotate bars with values
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.3f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom')

    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.show()

###############################################################################
def plot_modality_scores(modality_scores, concept_list, info=''):
    """
    Plots a bar chart of modality score for each concept.
    """

    modality_scores = {concept: score for concept, score in zip(concept_list, modality_scores)}
    modality_scores_sorted = dict(sorted(modality_scores.items(), key=lambda item: item[1], reverse=True))

    # Create bar plot
    plt.figure(figsize=(16, 5))
    bars = plt.bar(modality_scores_sorted.keys(), modality_scores_sorted.values(), color='skyblue')
    plt.xlabel('concepts')
    plt.ylabel('modality score')
    plt.title(info)
    
    # Dynamic y-axis limits centered around 0.5
    min_score = min(modality_scores_sorted.values())
    max_score = max(modality_scores_sorted.values())
    
    # Add 5% padding to actual range
    padding = (max_score - min_score) * 0.05
    y_min = min_score - padding
    y_max = max_score + padding
    
    # Ensure minimum range of [0.4, 0.6] centered at 0.5
    y_min = min(y_min, 0.4)
    y_max = max(y_max, 0.6)
    
    # Ensure symmetry around 0.5
    max_distance = max(0.5 - y_min, y_max - 0.5)
    y_min = 0.5 - max_distance
    y_max = 0.5 + max_distance
    
    plt.ylim(y_min, y_max)

    # Annotate bars with values
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.2f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom')

    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.show()