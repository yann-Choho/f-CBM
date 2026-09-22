import os
import json
import pandas as pd
from sklearn.metrics import pairwise_distances
import hdbscan
from umap import UMAP
import matplotlib.pyplot as plt

def cluster_and_visualize_topics(
    config,
    annotation: str,
    min_cluster_size: int = 2,
    random_state: int = 42,
    df_aug_train =None,
):
    """
    1) Charge le bon fichier de configuration en fonction de model_name et dataset.
    2) Charge et pré-traite df_aug_train selon le type d'annotation ('C3M' ou 'our_annotation').
    3) Extrait la matrice binaire de concepts, calcule la distance de Hamming, fait un clustering HDBSCAN,
       réassigne les points 'noise' ou les garde séparés (deux stratégies),
       construit les dictionnaires cluster→[concepts], affiche les embeddings UMAP, et sauvegarde les JSON.

    Arguments :
    - config            :  config file 
    - model_name        : 'bert-base-uncased' ou 'deberta-large'
    - annotation        : 'C3M' ou 'our_annotation'
    - min_cluster_size  : taille minimale pour HDBSCAN (défaut=2)
    - random_state      : graine aléatoire pour UMAP (défaut=42)
    """
    # --- C) Construction de la matrice binaire de concepts ---
    
    cols_to_drop = [col for col in ["text", "label"] if col in df_aug_train.columns]
    df_bin = df_aug_train.drop(columns=cols_to_drop)
    df_bin = df_bin.select_dtypes(exclude=["object"])
    
    df_transposed = df_bin.T  # chaque ligne = un concept binaire

    # --- D) Calcul de la matrice de distance ---
    if config.annotation != "cb_llm":
        # Données binaires : distance de Hamming
        dist_matrix = pairwise_distances(df_transposed, metric="hamming")
    else:
        # Cosine similarities : distance euclidienne
        dist_matrix = pairwise_distances(df_transposed, metric="euclidean")
        
    # --- E) Clustering HDBSCAN ---
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="precomputed")
    labels = clusterer.fit_predict(dist_matrix)

    # --- F) Stratégie 1 : réassigner les noise à leur cluster le plus proche ---
    # F1) Centroides (moyennes binaires) pour chaque cluster ≠ -1
    cluster_ids = [c for c in set(labels) if c != -1]
    centroids = {
        c: df_transposed.iloc[labels == c].mean(axis=0).values
        for c in cluster_ids
    }

    # F2) Réassignation des points noise (label == -1)
    labels_strat1 = labels.copy()
    noise_idx = [i for i, l in enumerate(labels) if l == -1]

    for i in noise_idx:
        point = df_transposed.iloc[i].values.reshape(1, -1)
        dists = {
            c: ((point - centroids[c])**2).sum() ** 0.5
            for c in cluster_ids
        }
        nearest = min(dists, key=dists.get)
        labels_strat1[i] = nearest

    # --- G) Stratégie 2 : garder le noise comme cluster séparé (-1) ---
    labels_strat2 = labels.copy()

    # --- H) Construction des dictionnaires cluster→liste de concepts ---
    def build_cluster_dict(labels_arr):
        d = {}
        for lbl in sorted(set(labels_arr)):
            concept_list = df_transposed.index[labels_arr == lbl].tolist()
            d[lbl] = concept_list
        return d

    cluster_dict_strat1 = build_cluster_dict(labels_strat1)
    cluster_dict_strat2 = build_cluster_dict(labels_strat2)

    # --- I) Visualisation UMAP pour les deux stratégies ---
    reducer = UMAP(metric="precomputed", random_state=random_state)
    embed = reducer.fit_transform(dist_matrix)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    for ax, (lbls, title) in zip(
        axes,
        [
            (labels_strat1, "Stratégie 1 : noise réassigné"),
            (labels_strat2, "Stratégie 2 : noise séparé")
        ]
    ):
        scatter = ax.scatter(embed[:, 0], embed[:, 1], c=lbls, s=80)
        for i, concept in enumerate(df_transposed.index):
            ax.text(embed[i, 0] + 0.002, embed[i, 1] + 0.002, concept, fontsize=7)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.grid(True)

        # Légende uniquement sur le deuxième subplot
        if title.startswith("Stratégie 2"):
            unique_labels = sorted(set(lbls))
            handles = [
                plt.Line2D([], [], marker="o", linestyle="", label=f"Cluster {u}")
                for u in unique_labels
            ]
            ax.legend(handles=handles, title="Clusters",
                      bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.show()

    # --- J) Affichage des dictionnaires dans la console ---
    print("Dictionnaire Stratégie 1 (noise réassigné) :")
    for cluster_id, concepts in cluster_dict_strat1.items():
        print(f"Cluster {cluster_id} : {concepts}\n")

    print("Dictionnaire Stratégie 2 (noise séparé) :")
    for cluster_id, concepts in cluster_dict_strat2.items():
        print(f"Cluster {cluster_id} : {concepts}\n")

    # --- K) Sauvegarde des dictionnaires au format JSON ---
    os.makedirs(config.path_to_output, exist_ok=True)

    filename1 = f"cluster_dict_strat_reassignation_{annotation}.json"
    filename2 = f"cluster_dict_strat_noise_cluster_{annotation}.json"

    path1 = os.path.join(config.path_to_output, filename1)
    path2 = os.path.join(config.path_to_output, filename2)

    # Convertir clés en string pour la sérialisation JSON
    cluster_dict_strat1_str = {str(k): v for k, v in cluster_dict_strat1.items()}
    cluster_dict_strat2_str = {str(k): v for k, v in cluster_dict_strat2.items()}

    with open(path1, "w", encoding="utf-8") as f:
        json.dump(cluster_dict_strat1_str, f, ensure_ascii=False, indent=4)

    with open(path2, "w", encoding="utf-8") as f:
        json.dump(cluster_dict_strat2_str, f, ensure_ascii=False, indent=4)

    print(f"Les dictionnaires JSON ont été enregistrés :\n- {path1}\n- {path2}")
