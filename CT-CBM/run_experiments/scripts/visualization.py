"""
Module de visualisation pour l'analyse des concepts et du coverage.
Contient les fonctions pour générer les graphiques de scores combinés et d'évolution du coverage.
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple


def clean_concept_name(name: str, model_name: str) -> str:
    """
    Nettoie les noms de concepts en retirant les préfixes et suffixes inutiles.
    
    Args:
        name: Nom du concept à nettoyer
        model_name: Nom du modèle (clip, blip, etc.)
    
    Returns:
        Nom nettoyé du concept
    """
    prefix = "concept_"
    name = name.replace(prefix, "").strip()
    return " ".join(name.split())


def plot_combined_scores_and_coverage(
    plot_df: pd.DataFrame,
    df_aug_train: pd.DataFrame,
    ordered_concepts: List[str],
    score_column: str = "combined_score_TCAVS",
    figsize: Tuple[int, int] = (18, 15),
    cbllm=False
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Affiche un graphique combiné avec les scores des concepts (barres) 
    et la couverture cumulative (courbe).
    
    Args:
        plot_df: DataFrame contenant les scores des concepts (indexé par 'concept')
        df_aug_train: DataFrame avec les données d'entraînement annotées
        ordered_concepts: Liste ordonnée des concepts à afficher
        score_column: Colonne de score à utiliser ('combined_score_TCAVS' ou 'combined_score_LIG')
        figsize: Taille de la figure
    
    Returns:
        Tuple contenant la figure et le tableau des pourcentages de coverage
    """
    # Calcul de la couverture cumulative
    counts = []
    concepts_iter = []
    
    for concept in ordered_concepts:
        concepts_iter.append(concept)
        if(cbllm):
            mean_activations = df_aug_train[concepts_iter].mean(axis=0)
            # coverage : nb de samples avec au moins une activation au dessus de la moyenne
            filtered_data = df_aug_train[concepts_iter][(df_aug_train[concepts_iter] >= mean_activations).any(axis=1)]
        else:
            filtered_data = df_aug_train[concepts_iter][
            df_aug_train[concepts_iter].eq(1).any(axis=1)]
        counts.append(len(filtered_data))
    
    # Calcul du pourcentage de lignes couvertes
    percentage = np.array([100 * c / df_aug_train.shape[0] for c in counts])
    positions = np.arange(len(ordered_concepts))
    
    # Création du graphique
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Barres : valeurs des scores
    values = plot_df.loc[ordered_concepts, score_column]
    ax1.bar(positions, values, color='skyblue', alpha=0.7, 
            label=f"Valeur moyenne des concepts ({score_column})")
    ax1.set_ylabel(score_column)
    ax1.set_xlabel("Concepts")
    ax1.set_title(f"Valeur moyenne des concepts et couverture cumulative ({score_column})")
    ax1.set_xticks(positions)
    ax1.set_xticklabels(ordered_concepts, rotation=45, ha='right')
    ax1.grid(axis="y", linestyle="--", alpha=0.7)
    
    # Axe secondaire : courbe cumulative
    ax2 = ax1.twinx()
    ax2.plot(positions, percentage, marker='o', color='darkorange', 
             label="Couverture cumulative")
    ax2.set_ylabel("Pourcentage de lignes avec au moins un concept (%)", 
                   color='darkorange')
    ax2.tick_params(axis='y', colors='darkorange')
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    
    # Légendes
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    
    plt.tight_layout()
    
    return fig, percentage


def plot_coverage_evolution(
    coverage_evolution: List[Tuple[List[str], float]],
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Affiche l'évolution du coverage cumulé en fonction du nombre de concepts sélectionnés.
    
    Args:
        coverage_evolution: Liste de tuples (concepts_sélectionnés, coverage)
        figsize: Taille de la figure
    
    Returns:
        Figure matplotlib
    """
    # Calcul du nombre cumulé de concepts
    n_concepts_selected = np.cumsum([len(concepts) for concepts, _ in coverage_evolution])
    coverage_values = [cov for _, cov in coverage_evolution]
    
    # Création du graphique
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(n_concepts_selected, coverage_values, marker='o', linestyle='-')
    ax.set_xlabel("Nombre de concepts sélectionnés")
    ax.set_ylabel("Coverage cumulé (%)")
    ax.set_title("Évolution du coverage en sélection round-robin par cluster")
    ax.grid(True)
    
    plt.tight_layout()
    
    return fig


def print_coverage_steps(coverage_evolution: List[Tuple[List[str], float]]) -> None:
    """
    Affiche les étapes de sélection des concepts avec le coverage associé.
    
    Args:
        coverage_evolution: Liste de tuples (concepts_sélectionnés, coverage)
    """
    for idx, (concepts, cov) in enumerate(coverage_evolution):
        print(f"Step {idx+1}: Concepts {concepts} → Coverage {cov:.2f}%")