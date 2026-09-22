"""
Module pour le calcul des scores combinés par cluster et la sélection de concepts
basée sur le coverage cumulatif.

# 
# Extension du pipeline_dispatcher pour l'analyse de coverage par cluster.

# Ce module ajoute une étape d'analyse post-scoring qui :
# 1. Calcule le coverage cumulatif des concepts
# 2. Organise les concepts par cluster
# 3. Effectue une sélection round-robin multi-cluster
# 4. Génère les visualisations associées
# 

"""

import json
import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from visualization import (
    clean_concept_name,
    plot_combined_scores_and_coverage,
    plot_coverage_evolution,
    print_coverage_steps
)

def load_cluster_assignments(
    path_to_output: str,
    clean_concept_name_func,
    config
) -> Dict[int, List[str]]:
    """
    Charge les dictionnaires de clusters et applique le nettoyage des noms.
    
    Args:
        path_to_output: Chemin vers le dossier contenant les fichiers de clusters
        clean_concept_name_func: Fonction de nettoyage des noms de concepts
    
    Returns:
        Dictionnaire {cluster_id: [concepts]}
    """
    path_reassignation = Path(path_to_output) / f"cluster_dict_strat_reassignation_{config.annotation}.json"
    
    with open(path_reassignation, 'r') as f:
        cluster_dict = json.load(f)
    
    # Appliquer le nettoyage
    cluster_dict_clean = {
        k: [clean_concept_name_func(name) for name in v] 
        for k, v in cluster_dict.items()
    }
    
    return cluster_dict_clean


def assign_concepts_to_clusters(
    plot_df: pd.DataFrame,
    cluster_dict: Dict[int, List[str]]
) -> pd.DataFrame:
    """
    Assigne chaque concept à son cluster dans le DataFrame.
    
    Args:
        plot_df: DataFrame contenant les concepts et leurs scores
        cluster_dict: Dictionnaire {cluster_id: [concepts]}
    
    Returns:
        DataFrame avec une nouvelle colonne 'cluster'
    """
    def get_cluster(concept: str) -> int:
        for cluster_id, concepts in cluster_dict.items():
            if concept in concepts:
                return int(cluster_id)
        return -1  # Concept non assigné
    
    plot_df = plot_df.copy()
    plot_df['cluster'] = plot_df['concept'].apply(get_cluster)
    
    return plot_df


def build_cluster_groups(
    plot_df: pd.DataFrame,
    score_column: str = "combined_score_TCAVS"
) -> Dict[int, List[str]]:
    """
    Construit un dictionnaire de concepts triés par score pour chaque cluster.
    
    Args:
        plot_df: DataFrame avec concepts, scores et clusters (indexé par 'concept')
        score_column: Colonne de score à utiliser pour le tri
    
    Returns:
        Dictionnaire {cluster_id: [concepts triés par score décroissant]}
    """
    cluster_groups = {}
    
    for cluster_id in sorted(set(plot_df['cluster'].tolist())):
        concepts_sorted = (
            plot_df
            .loc[plot_df["cluster"] == cluster_id]
            .sort_values(score_column, ascending=False)
            .index
            .tolist()
        )
        cluster_groups[cluster_id] = concepts_sorted
    
    return cluster_groups


def round_robin_selection_with_coverage(
    cluster_groups: Dict[int, List[str]],
    df_aug_train: pd.DataFrame,
    cbllm=False
) -> List[Tuple[List[str], float]]:
    """
    Sélectionne les concepts selon une stratégie round-robin par cluster
    et calcule le coverage cumulatif à chaque étape.
    
    Args:
        cluster_groups: Dictionnaire {cluster_id: [concepts triés]}
        df_aug_train: DataFrame avec les annotations binaires des concepts
    
    Returns:
        Liste de tuples (concepts_sélectionnés_cette_étape, coverage_cumulé)
    """
    total_samples = df_aug_train.shape[0]
    working = {c: cluster_groups[c].copy() for c in cluster_groups}
    
    coverage_evolution = []
    selected_concepts = []
    
    while any(working[c] for c in working):
        current_selection = []
        
        # Sélectionner un concept par cluster (round-robin)
        for cluster_id in sorted(working):
            if working[cluster_id]:
                concept = working[cluster_id].pop(0)
                current_selection.append(concept)
                selected_concepts.append(concept)
        
        # Calcul du coverage après ajout de ces concepts
        if(cbllm):

            mean_activations = df_aug_train[selected_concepts].mean(axis=0)
            
            # coverage : nb de samples avec au moins une activation au dessus de la moyenne
            filtered_data = df_aug_train[selected_concepts][(df_aug_train[selected_concepts] >= mean_activations).any(axis=1)]

            coverage = 100 * len(filtered_data) / total_samples

        else:

            mask = df_aug_train[selected_concepts].eq(1).any(axis=1)
            coverage = mask.sum() * 100 / total_samples
        
        coverage_evolution.append((current_selection.copy(), coverage))
    
    return coverage_evolution


def compute_ordered_concepts_coverage(
    plot_df: pd.DataFrame,
    df_aug_train: pd.DataFrame,
    score_column: str = "combined_score_TCAVS",
    cbllm=False
) -> Tuple[List[str], Dict[str, float]]:
    """
    Calcule le coverage cumulatif des concepts triés par score décroissant.
    
    Args:
        plot_df: DataFrame contenant les concepts et leurs scores
        df_aug_train: DataFrame avec les annotations binaires
        score_column: Colonne de score à utiliser pour le tri
    
    Returns:
        Tuple (liste_ordonnée_concepts, dict {concept: coverage_cumulé})
    """
    # Tri des concepts par score décroissant
    ordered_concepts = (
        plot_df
        .sort_values(by=score_column, ascending=False)['concept']
        .tolist()
    )
    
    # Calcul du coverage cumulatif
    total_samples = df_aug_train.shape[0]
    concepts_iter = []
    percentage = []
    
    for concept in ordered_concepts:

        concepts_iter.append(concept)

        if(cbllm):
            
            # moyenne de toutes les activations
            #mean_activations = np.mean(df_aug_train[[column for column in df_aug_train.columns if column.startswith(concept_)]].values)
            # moyenne des activations par concept choisis
            mean_activations = df_aug_train[concepts_iter].mean(axis=0)
            
            # coverage : nb de samples avec au moins une activation au dessus de la moyenne
            filtered_data = df_aug_train[concepts_iter][(df_aug_train[concepts_iter] >= mean_activations).any(axis=1)]

        else:

            filtered_data = df_aug_train[concepts_iter][
                df_aug_train[concepts_iter].eq(1).any(axis=1)
            ]
            
        cov = 100 * len(filtered_data) / total_samples

        percentage.append(cov)
    
    score_dict = dict(zip(ordered_concepts, percentage))
    
    return ordered_concepts, score_dict


def save_coverage_results(
    ordered_concepts: List[str],
    score_dict: Dict[str, float],
    coverage_evolution: List[Tuple[List[str], float]],
    config,
    method_suffix: str = "TCAVS"
) -> None:
    """
    Sauvegarde les résultats de coverage dans des fichiers JSON et pickle.
    
    Args:
        ordered_concepts: Liste ordonnée des concepts
        score_dict: Dictionnaire {concept: coverage}
        coverage_evolution: Liste des étapes de sélection round-robin
        config: Objet de configuration contenant les chemins
        method_suffix: Suffixe pour identifier la méthode (TCAVS, LIG, etc.)
    """
    base_path = (
        f"{config.SAVE_PATH}/blue_checkpoints/{config.model_name}/cavs"
        f"/{config.cavs_type}"
    )
    
    # Sauvegarde du coverage par concept (ordre simple)
    path_coverage_json = (
        f"{base_path}/sorted_macro_concepts_coverage_"
        f"{config.annotation}_{config.agg_mode}_{config.agg_scope}_{method_suffix}.json"
    )
    
    with open(path_coverage_json, 'w') as f:
        json.dump(score_dict, f, ensure_ascii=False, indent=4)
    
    print(f"✓ Fichier sauvegardé : {path_coverage_json}")
    
    # Sauvegarde du coverage par étape (sélection round-robin multi-cluster)
    path_coverage_pkl = (
        f"{base_path}/sorted_macro_concepts_coverage_MJ_"
        f"{config.annotation}_{config.agg_mode}_{config.agg_scope}_{method_suffix}.pkl"
    )
    
    with open(path_coverage_pkl, "wb") as f:
        pickle.dump(coverage_evolution, f)
    
    print(f"✓ Fichier sauvegardé : {path_coverage_pkl}")


def compute_and_save_coverage_analysis(
    config,
    df_aug_train: pd.DataFrame,
    clean_concept_name_func,
    score_column: str = "combined_score_TCAVS"
) -> Tuple[pd.DataFrame, List[Tuple[List[str], float]]]:
    """
    Pipeline complet pour calculer et sauvegarder l'analyse de coverage.
    
    Cette fonction :
    1. Charge les scores combinés des concepts
    2. Charge les assignations aux clusters
    3. Calcule le coverage par ordre de score décroissant
    4. Calcule le coverage par sélection round-robin multi-cluster
    5. Sauvegarde tous les résultats
    
    Args:
        config: Objet de configuration
        df_aug_train: DataFrame avec annotations binaires
        clean_concept_name_func: Fonction de nettoyage des noms
        score_column: Colonne de score à utiliser
    
    Returns:
        Tuple (plot_df avec clusters, coverage_evolution)
    """
    # 1. Charger les scores combinés
    path_plot_df = (
        f"{config.SAVE_PATH}/blue_checkpoints/{config.model_name}/cavs"
        f"/{config.cavs_type}/combined_score_concept_"
        f"{config.annotation}_{config.agg_mode}_{config.agg_scope}.csv"
    )
    plot_df = pd.read_csv(path_plot_df)
    
    # 2. Nettoyer les noms de colonnes du DataFrame d'entraînement
    df_aug_train.rename(
        columns=lambda x: clean_concept_name_func(x), 
        inplace=True
    )
    
    # 3. Calculer le coverage par ordre simple (score décroissant)
    ordered_concepts, score_dict = compute_ordered_concepts_coverage(
        plot_df, df_aug_train, score_column, True if config.annotation == "cb_llm" else False
    )
    
    # 4. Charger et assigner les clusters
    cluster_dict = load_cluster_assignments(
        config.path_to_output, 
        clean_concept_name_func,
        config
    )
    plot_df = assign_concepts_to_clusters(plot_df, cluster_dict)
    
    # 5. Préparer le DataFrame pour la sélection round-robin
    plot_df = plot_df.set_index("concept")
    
    # 6. Construire les groupes de concepts par cluster
    cluster_groups = build_cluster_groups(plot_df, score_column)
    
    # 7. Sélection round-robin avec calcul du coverage
    coverage_evolution = round_robin_selection_with_coverage(
        cluster_groups, df_aug_train, True if config.annotation == "cb_llm" else False
    )
    
    # 8. Sauvegarder les résultats
    method_suffix = score_column.split("_")[-1]  # TCAVS ou LIG
    save_coverage_results(
        ordered_concepts, 
        score_dict, 
        coverage_evolution, 
        config, 
        method_suffix
    )
    
    return plot_df, coverage_evolution


def run_coverage_analysis_pipeline(
    config,
    df_aug_train: pd.DataFrame,
    score_column: str = "combined_score_TCAVS",
    show_plots: bool = True
) -> Tuple[pd.DataFrame, List[Tuple[List[str], float]]]:
    """
    Pipeline complet pour l'analyse de coverage des concepts.
    
    Cette fonction est appelée après le calcul des scores TCAVS/LIG et effectue :
    - Le calcul du coverage par concept
    - La sélection round-robin par cluster
    - La génération des visualisations
    - La sauvegarde des résultats
    
    Args:
        config: Objet de configuration du pipeline
        df_aug_train: DataFrame avec les annotations binaires des concepts
        score_column: Colonne de score à utiliser ('combined_score_TCAVS' ou 'combined_score_LIG')
        show_plots: Si True, affiche les graphiques
    
    Returns:
        Tuple (plot_df avec clusters, coverage_evolution)
    
    Note:
        Cette fonction ne doit PAS être appelée en mode cb_llm car le coverage
        n'est pas défini dans ce contexte (pas d'annotations binaires).
    """
    # Fonction de nettoyage des noms de concepts
    def clean_name(name: str) -> str:
        return clean_concept_name(name, config.model_name)
    
    print("\n" + "="*80)
    print(f"ANALYSE DE COVERAGE PAR CLUSTER ({score_column})")
    print("="*80)
    
    # 1. Calculer et sauvegarder l'analyse complète
    print("\n[1/3] Calcul du coverage et sélection round-robin...")
    # if(config.annotation == "cb_llm"):
    #     plot_df = compute_and_save_coverage_analysis_cbllm(
    #         config=config,
    #         df_aug_train=df_aug_train,
    #         clean_concept_name_func=clean_name,
    #         score_column=score_column
    #     )
    # else:
    plot_df, coverage_evolution = compute_and_save_coverage_analysis(
                config=config,
                df_aug_train=df_aug_train,
                clean_concept_name_func=clean_name,
                score_column=score_column
            )
    
    # 2. Afficher les étapes de sélection
    print("\n[2/3] Étapes de sélection round-robin par cluster :")
    print_coverage_steps(coverage_evolution)
    
    # 3. Générer les visualisations si demandé
    if show_plots:
        print("\n[3/3] Génération des visualisations...")
        
        # Graphique 1 : Scores + coverage cumulatif (ordre simple)
        ordered_concepts, _ = compute_ordered_concepts_coverage(
            plot_df.reset_index(), 
            df_aug_train, 
            score_column,
            True if config.annotation == "cb_llm" else False
        )
        
        plot_df_for_viz = plot_df.copy()
        if 'concept' not in plot_df_for_viz.columns:
            plot_df_for_viz = plot_df_for_viz.reset_index()
            plot_df_for_viz = plot_df_for_viz.set_index('concept', drop=False)
        
        fig1, percentage = plot_combined_scores_and_coverage(
            plot_df=plot_df_for_viz,
            df_aug_train=df_aug_train,
            ordered_concepts=ordered_concepts,
            score_column=score_column,
            cbllm=True if config.annotation == "cb_llm" else False
        )
        fig1.show()
        
        # Graphique 2 : Évolution du coverage (round-robin)
        fig2 = plot_coverage_evolution(coverage_evolution)
        fig2.show()
        
        print("✓ Visualisations générées")
    
    print("\n" + "="*80)
    print("ANALYSE TERMINÉE")
    print("="*80 + "\n")
    
    return plot_df, coverage_evolution


# ============================================================================
# FONCTION D'INTÉGRATION DANS LE DISPATCHER PRINCIPAL
# ============================================================================
def do_coverage_analysis(
    config,
    df_aug_train: pd.DataFrame,
    TCAVS_or_LIG
) -> dict:
    """
    Intègre l'analyse de coverage dans le pipeline principal.
    
    Cette fonction peut être appelée après l'étape de scoring (Step 5)
    pour analyser le coverage des concepts selon différentes méthodes.
    
    Args:
        config: Objet de configuration
        df_aug_train: DataFrame avec annotations binaires
        TCAVS_or_LIG: Si True, lance l'analyse pour TCAVS et LIG
    
    Returns:
        Dictionnaire contenant les résultats pour chaque méthode
    
    """
    if config.cb_llm_mode:
        print("⚠️  Mode cb_llm détecté : l'analyse de coverage est ignorée")
        print("   (le coverage n'est pas défini pour cb_llm)")
        return {}
    
    results = {}
    
    # Analyse avec TCAVS
    if TCAVS_or_LIG ==  'both' or TCAVS_or_LIG == 'TCAVS':  # TCAVS par défaut
        print("\n🔍 Analyse avec combined_score_TCAVS...")
        plot_df_tcavs, coverage_tcavs = run_coverage_analysis_pipeline(
            config=config,
            df_aug_train=df_aug_train,
            score_column="combined_score_TCAVS",
            show_plots=True
        )
        results['TCAVS'] = {
            'plot_df': plot_df_tcavs,
            'coverage_evolution': coverage_tcavs
        }
    
    # Analyse avec LIG
    if TCAVS_or_LIG == 'both' or TCAVS_or_LIG == 'LIG':
        print("\n🔍 Analyse avec combined_score_LIG...")
        plot_df_lig, coverage_lig = run_coverage_analysis_pipeline(
            config=config,
            df_aug_train=df_aug_train,
            score_column="combined_score_LIG",
            show_plots=True
        )
        results['LIG'] = {
            'plot_df': plot_df_lig,
            'coverage_evolution': coverage_lig
        }
    
    return results
