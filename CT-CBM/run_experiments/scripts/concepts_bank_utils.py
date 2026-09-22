## ---------------------------  Topic modelling pour creer la base de concepts ---------------------------
# A JOUR

from transformers import pipeline
import sys
sys.path.append('./run_experiments')
sys.path.append('./run_experiments/models')

import ast
import re
import pandas as pd
import joblib
import json
from hdbscan import flat
import torch
import hdbscan
import umap 
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import numpy as np 
from sklearn.metrics.pairwise import euclidean_distances



def safe_literal_eval(val):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return val  # Ou renvoyer une valeur par défaut comme une liste vide


def prepare_data(df):
    # Copier le dataset
    df_ = df.copy()
    
    # Vérifier si la colonne 'text' existe dans le DataFrame
    # if 'text' in df_.columns:
    #     # Tronquer la colonne 'text' à 3000 caractères uniquement si c'est une chaîne de caractères
    #     df_['text'] = df_['text'].apply(lambda x: x[:3000] if isinstance(x, str) else x)
    # else:
    #     print("La colonne 'text' n'existe pas dans le DataFrame")
    
    print(df_.head(3))  # Utiliser head() pour ne pas imprimer l'intégralité du DataFrame
    return df_



def extract_topics(text, discovery_model, discovery_tokenizer, device = None):
    """
    Génère un macro concept (unique représentant) pour une liste de concepts en utilisant un modèle de génération de texte.
    
    Args:
    - text (list): La liste des concepts à résumer.
    - discovery_model: Le modèle de génération de texte (comme GPT).
    - discovery_tokenizer: Le tokenizer correspondant au modèle.
    - device: L'appareil sur lequel le modèle sera exécuté (CPU ou GPU).
    
    Returns:
    - answer (str): La réponse générée par le modèle, c'est-à-dire le macro concept.
    """

    # Préparation du prompt
    concepts_list_1 = "As cities expand and populations grow, there is a growing tension between development and the need to preserve historical landmarks. Citizens and authorities often clash over the balance between progress and cultural heritage."
    micro_concept_1 = "[urban development, cultural heritage, conflict]"
    
    concepts_list_2 = "Recent breakthroughs in neuroscience are shedding light on the complexities of human cognition. Researchers are particularly excited about the potential to better understand decision-making processes and emotional regulation in the brain."
    micro_concept_2 = "[neuroscience, human cognition, decision-making, emotional regulation]"
    

    preprompt = '''You are presented with several parts of speech.
    Identify only the main topics in this text. Respond with topic in list format like the examples in a very concise way using as few words as possible'''
    
    messages = [{"role": "user", "content": preprompt + "\nExample\n" + str(concepts_list_1)}]
    messages.append({"role": "assistant", "content": "Topics: "+ str(micro_concept_1)+"<eos>"})  
    
    messages.append({"role": "user", "content":str(concepts_list_2)})
    messages.append({"role": "assistant", "content": "Topics: "+ str(micro_concept_2)+"<eos>"})  
     
    # Création du message avec la liste de concepts
    messages.append({"role": "user", "content":str(text)})
    messages.append({"role": "assistant", "content":"Topics: ["})
    
    # Encoder le message dans un format utilisable par le modèle
    encoded_input = discovery_tokenizer.apply_chat_template(messages, return_tensors='pt')
    encoded_input = torch.reshape(encoded_input[0][:-2], (1, encoded_input[0][:-2].shape[0]))
    
    # Calculer la longueur de l'entrée pour obtenir uniquement la sortie générée
    len_input = encoded_input.shape[1]
    
    # Générer le macro concept en utilisant le modèle
    outputs = discovery_model.generate(
        encoded_input.to(device), 
        max_new_tokens=50, 
        do_sample=True, 
        num_beams=2, 
        no_repeat_ngram_size=2, 
        early_stopping=True, 
        temperature=1
    )
    
    # Décoder et formater la réponse
    answer = "[" + discovery_tokenizer.decode(outputs[0][len_input:], skip_special_tokens=True)
    print("Generated Micro Concept: ", answer)
    return answer
    # return {"topics":extract_values(answer)} 

def extract_values(d):
    """
    Extrait les valeurs d'une chaîne de caractères au format [valeur1, valeur2, ...] et retourne une liste de ces valeurs.
    Si aucun format spécifique n'est détecté, renvoie simplement une liste des mots dans la chaîne.
    
    Args:
    - d (str): Le texte d'entrée sous forme de chaîne.
    
    Returns:
    - list: Une liste des valeurs extraites ou, à défaut, des mots trouvés dans la chaîne.
    """
    try:
        if isinstance(d, str):
            # Rechercher des valeurs entre crochets et les séparer par des virgules
            matches = re.findall(r'\[([^\]]+)\]', d)
            if matches:
                # Séparer les éléments par des virgules et supprimer les espaces supplémentaires
                result = [item.strip() for item in matches[0].split(',')]
                
                # Vérifier que tous les éléments sont non vides après nettoyage
                result = [item for item in result if item]
                
                # Retourner la liste si elle contient des éléments valides
                if result:
                    return result
            
            # Si aucun motif particulier n'est trouvé, on peut retourner simplement les mots dans la chaîne
            return re.findall(r'\b\w+\b', d)
        else:
            print(f"Type inattendu: attendu str, reçu {type(d)}")
    except Exception as e:
        print(f"Erreur inattendue: {e}")
    
    # Retourner une liste vide si aucune extraction ne réussit
    return []

# Appliquer la fonction à la colonne 'topics' pour obtenir une nouvelle colonne avec les valeurs extraites
def process_extracted_topics(df_with_topics, discovery_model, discovery_tokenizer, device = torch.device("cuda"), config = None):
    actual_device = config.device if config is not None else device

    df_with_topics['topics'] = df_with_topics['text'].apply(lambda text: extract_topics(text, discovery_model, discovery_tokenizer, device = actual_device))
    df_with_topics['extracted_topics'] = df_with_topics['topics'].apply(extract_values)
    return df_with_topics

# Enregistrement des résultats
def save_results(df_with_topics, filename):
    df_with_topics.to_csv(f'{config.SAVE_PATH_CONCEPTS}/{filename}.csv', index = False)
    print(f"Results saved to others/{filename}.csv")

# -----> df_with_topics_v1.csv
# ----> RESULT in list format per line in the dataframe so we need to extract it


# Créer une liste unique de tous les sujets sans doublons
def create_unique_topic_list(df_with_topics):
    all_topics = [topic for topics in df_with_topics['extracted_topics'].apply(safe_literal_eval) for topic in topics]
    # all_topics = set(topic for topics in df_with_topics['extracted_topics'] for topic in topics)
    all_topics = list(set(all_topics))
    return all_topics

def topic_with_count_df(df_with_topics, config = None):
    from collections import Counter

    # Flatten the list of topics and count occurrences
    all_topics = [topic for topics in df_with_topics['extracted_topics'].apply(safe_literal_eval) for topic in topics]
    topic_counts = Counter(all_topics)
    
    # Convert to a list of tuples (topic, count)
    unique_topics_with_counts = list(topic_counts.items())
    df_topics = pd.DataFrame(unique_topics_with_counts, columns=['Topic', 'Count'])

    # sort from most to least frequent
    df_topics = df_topics.sort_values(by='Count', ascending=False)

    df_topics.to_csv(f'{config.SAVE_PATH_CONCEPTS}/topics_counts.csv', index=False)
    print("Les résultats ont été enregistrés dans 'topics_counts.csv'.")
    return df_topics
    

# ------------------------ ADD THE LOGIC OF MACRO CONCEPT BELOW 
def create_macro_concepts_pipeline(df_path, save_path, discovery_model, discovery_tokenizer, n_clusters=10, model_name='all-mpnet-base-v2',  config = None):
    # Charger le fichier CSV initial avec les topics extraits
    df_ = pd.read_csv(df_path)

    # Étape 1: Créer une liste unique de tous les concepts (topics)
    topics_set = create_unique_topic_list(df_)  # Liste unique de tous les topics
    concepts = list(topics_set)

    # Étape 2: Obtenir les embeddings de tous les concepts
    mpnet_model = SentenceTransformer(model_name)
    embeddings = torch.stack([torch.Tensor(mpnet_model.encode(concept)) for concept in concepts])
    embeddings_np = embeddings.cpu().numpy()

    # Étape 3: Réduire les dimensions des embeddings avec UMAP
    reducer = umap.UMAP(n_components=5, random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings_np)


    # Étape 4: Clusterisation avec HDBSCAN
    clusterer = flat.HDBSCAN_flat(embeddings_2d, n_clusters=n_clusters, prediction_data=True)
    clusters = flat.approximate_predict_flat(clusterer, embeddings_2d, n_clusters=n_clusters)

    # Sauvegarder UMAP et DBSCAN
    joblib.dump(reducer, f'{save_path}/umap_model.pkl')
    joblib.dump(clusterer, f'{save_path}/dbscan_model.pkl')
    print("UMAP, DBSCAN, and clusters saved.")

    # Étape 5: Créer un DataFrame pour stocker les clusters
    df_clusters = pd.DataFrame({'concept': concepts, 'cluster': clusters[0]})
    df_clusters.to_csv(f"{save_path}/df_cluster.csv", index=False)

    # Étape 6: Fonction pour obtenir les représentants d'un cluster (jusqu'à 15 concepts les plus proches du centroid)
    def get_top_n_representatives(embeddings_np, df_clusters, top_n=15):
        cluster_representatives = {}
        number_to_centroids = {}

        for cluster in df_clusters['cluster'].unique():
            if cluster != -1:  # Ignorer les bruits
                cluster_indices = df_clusters[df_clusters['cluster'] == cluster].index
                cluster_embeddings = embeddings_np[cluster_indices]
                centroid = np.mean(cluster_embeddings, axis=0)
                distances = euclidean_distances(cluster_embeddings, centroid.reshape(1, -1)).flatten()
                # Obtenir les indices des 15 concepts les plus proches (ou moins si < 15 concepts)
                sorted_indices = np.argsort(distances)[:top_n]
                closest_concepts = df_clusters.iloc[cluster_indices[sorted_indices]]['concept'].tolist()
                cluster_representatives[cluster] = closest_concepts
                # print('type',type(cluster))
                # print('cluster', cluster)
                number_to_centroids[int(cluster)] = centroid.tolist()

        with open(f'{save_path}/number_to_centroids.json', 'w') as f:
                json.dump(number_to_centroids, f)

        return cluster_representatives

    # Obtenir les représentants de chaque cluster
    cluster_representatives = get_top_n_representatives(embeddings_2d, df_clusters)
    
    # Étape 7: Générer un macro concept unique pour chaque cluster
    def generate_macro_concept(concepts_list, discovery_model, discovery_tokenizer, device = torch.device("cuda")):
        """
        Génère un macro concept (unique représentant) pour une liste de concepts en utilisant un modèle de génération de texte.
        
        Args:
        - concepts_list (list): La liste des concepts à résumer.
        - discovery_model: Le modèle de génération de texte (comme GPT).
        - discovery_tokenizer: Le tokenizer correspondant au modèle.
        - device: L'appareil sur lequel le modèle sera exécuté (CPU ou GPU).
        
        Returns:
        - answer (str): La réponse générée par le modèle, c'est-à-dire le macro concept.
        """
        
        # Préparation du prompt
        concepts_list_1 = ["piano", "guitar", "saxophone", "violin", "cheyenne", "drum"]
        macro_concept_1 = "musical instrument"
        
        concepts_list_2 = ["football", "basketball", "baseball", "tennis", "badmington", "soccer"]
        macro_concept_2 = "sport"
        
        concepts_list_3 = ["lion", "tiger", "cat", "pumas", "panther", "leopard"]
        macro_concept_3 = "feline-type animal"
 
        preprompt = '''You are presented with several parts of speech.
        Summarise what these parts of speech have in common in a very concise way using as few words as possible'''
        
        messages = [{"role": "user", "content": preprompt + "\nExample\n" + str(concepts_list_1)}]
        messages.append({"role": "assistant", "content": "Summarization: "+macro_concept_1+"<eos>"})  
        
        messages.append({"role": "user", "content":str(concepts_list_2)})
        messages.append({"role": "assistant", "content": "Summarization: "+macro_concept_2+"<eos>"})  
        
        messages.append({"role": "user", "content":str(concepts_list_3)})
        messages.append({"role": "assistant", "content": "Summarization: "+macro_concept_3+"<eos>"})  
        
        # Création du message avec la liste de concepts
        messages.append({"role": "user", "content":str(concepts_list)})
        messages.append({"role": "assistant", "content":"Summarization: "})

        # Encoder le message dans un format utilisable par le modèle
        encoded_input = discovery_tokenizer.apply_chat_template(messages, return_tensors='pt')
        encoded_input = torch.reshape(encoded_input[0][:-2], (1, encoded_input[0][:-2].shape[0]))
        
        # Calculer la longueur de l'entrée pour obtenir uniquement la sortie générée
        len_input = encoded_input.shape[1]
        
        # Générer le macro concept en utilisant le modèle
        outputs = discovery_model.generate(
            encoded_input.to(device), 
            max_new_tokens=50, 
            do_sample=True, 
            num_beams=2, 
            no_repeat_ngram_size=2, 
            early_stopping=True, 
            temperature=1
        )
        
        # Décoder et formater la réponse
        answer = discovery_tokenizer.decode(outputs[0][len_input:], skip_special_tokens=True)
        print("Generated Macro Concept: ", answer)
        
        return answer

    with open(f'{save_path}/number_to_centroids.json', 'r') as f:
        number_to_centroids_ = json.load(f)

    # Convertir les centroïdes de listes Python en tableaux NumPy
    # for cluster, centroid in number_to_centroids_.items():
    #     number_to_centroids_[cluster] = np.array(centroid)


    # Générer un macro concept unique pour chaque cluster
    macro_concepts = {}
    macro_to_centroid = {}
    for cluster, concepts_list in cluster_representatives.items():
        # get the summary of cluster via prompt
        macro_concept = generate_macro_concept(concepts_list, discovery_model, discovery_tokenizer)

        # Save the macro concept summary : centroid_np_vector in a json file
        macro_concepts[str(cluster)] = macro_concept
        centroid_vector = number_to_centroids_[str(cluster)]
        macro_to_centroid[macro_concept] = centroid_vector
        print(f"Cluster {cluster}: {macro_concept}")
        print(concepts_list)

    with open(f'{save_path}/macro_to_centroid.json', 'w') as f:
        json.dump(macro_to_centroid, f)

    with open(f'{save_path}/number_to_macro.json', 'w') as f:
        json.dump(macro_concepts, f)
    

    # Étape 7: Créer une correspondance entre chaque concept et son macro concept
    concept_to_macro = {}
    for cluster, macro_concept in macro_concepts.items():
        cluster_concepts = df_clusters[df_clusters['cluster'] == int(cluster)]['concept'].tolist() # attention aux int et str (cluster)
        for concept in cluster_concepts:
            concept_to_macro[concept] = macro_concept

    # Sauvegarder la correspondance concept -> macro concept en JSON
    with open(f'{save_path}/concept_to_macro.json', 'w') as f:
        json.dump(concept_to_macro, f)

    # Étape 8: Associer les macro concepts à chaque texte dans le DataFrame original
    def associate_macro_concepts(row, concept_to_macro):
        extracted_topics = ast.literal_eval(row['extracted_topics'])  # Transformer en liste si nécessaire
        macro_concepts = set()
        for topic in extracted_topics:
            if topic in concept_to_macro:
                macro_concepts.add(concept_to_macro[topic])
        return list(macro_concepts)

    df_['macro_concepts'] = df_.apply(associate_macro_concepts, axis=1, concept_to_macro=concept_to_macro)

    # Étape 9: Extraire les macro concepts uniques et créer une colonne dummy pour chaque macro concept
    unique_macro_concepts = set(concept for concepts in df_['macro_concepts'] for concept in concepts)

    # Construire un dictionnaire avec toutes les nouvelles colonnes
    new_columns = {f'dummy_{macro_concept}': df_['macro_concepts'].apply(lambda x: 1 if macro_concept in x else 0) for macro_concept in unique_macro_concepts}

    # Convertir en DataFrame et concaténer avec l'original
    df_ = pd.concat([df_, pd.DataFrame(new_columns)], axis=1)

    # Étape 10: Sauvegarder le DataFrame mis à jour avec les macro concepts et les colonnes dummy
    df_.to_csv(f"{save_path}/df_with_topics_v3.csv", index=False)
    print(f"Results saved to {save_path}/df_with_topics_v3.csv")

    # Sélection des colonnes 'text', 'label' et celles qui commencent par 'dummy_'
    colonnes_a_garder = [col for col in df_.columns if col.startswith('dummy_') or col in ['text', 'label']]
    df_filtre = df_[colonnes_a_garder]

    # Sauvegarde du nouveau DataFrame filtré
    df_filtre.to_csv(f"{save_path}/df_with_topics_v4.csv", index=False)
    print(f"Results saved to {save_path}/df_with_topics_v4.csv")

# ------------------------ LABEL TEST AND VAL DATA WITH MACRO CONCEPT
# input : df_with_topics_v2_{dataset} ,dataset = val/test
# output : df_with_topics_v4_{dataset} , dataset = val/test

def create_macro_concepts_pipeline_v2(df_path, save_path, model_name='all-mpnet-base-v2', config=None, reducer=None, clusterer=None, dataset = 'test'):
    """ Labelised test or val data with macro concepts 
    Args : df_with_topics_v2_{dataset} ,dataset = val/test
    Output : df_with_topics_v4_{dataset} , dataset = val/test
    """
    # Charger le fichier CSV initial avec les topics extraits
    df_ = pd.read_csv(df_path)

    # Étape 1: Créer une liste unique de tous les concepts (topics)
    topics_set = create_unique_topic_list(df_)  # Liste unique de tous les topics
    concepts = list(topics_set)

    # Étape 2: Obtenir les embeddings de tous les concepts
    mpnet_model = SentenceTransformer(model_name)
    embeddings = torch.stack([torch.Tensor(mpnet_model.encode(concept)) for concept in concepts])
    embeddings_np = embeddings.cpu().numpy()

    # Étape 3: Réduire les dimensions des embeddings avec le modèle UMAP chargé
    if reducer is None:
        raise ValueError("Le modèle UMAP (reducer) n'a pas été fourni.")
    embeddings_2d = reducer.transform(embeddings_np)

    # Étape 4: Clusterisation avec le modèle HDBSCAN chargé
    if clusterer is None:
        raise ValueError("Le modèle HDBSCAN (clusterer) n'a pas été fourni.")
    clusters = flat.approximate_predict_flat(clusterer, embeddings_2d)

    # Étape 5: Créer un DataFrame pour stocker les clusters
    df_clusters = pd.DataFrame({'concept': concepts, 'cluster': clusters[0]})
    # prendre que les different de -1
    df_clusters = df_clusters[df_clusters['cluster'] != -1]
    df_clusters.to_csv(f"{save_path}/df_cluster_{dataset}.csv", index=False)
    

    # Charger le dictionnaire number_to_macro
    with open(f'{save_path}/number_to_macro.json', 'r') as f:
        number_to_macro = json.load(f)

    # Étape 6: Associer les clusters aux textes dans le DataFrame original
    def associate_macro_concepts(row, concept_to_macro):
        extracted_topics = ast.literal_eval(row['extracted_topics'])  # Transformer en liste si nécessaire
        macro_concepts = set()
        for topic in extracted_topics:
            if topic in concept_to_macro:
                macro_concepts.add(concept_to_macro[topic])
        return list(macro_concepts)

    # Construire une correspondance entre chaque concept et son macro-concept
    concept_to_macro = {row['concept']: number_to_macro[str(row['cluster'])] for _, row in df_clusters.iterrows()}
    df_['macro_concepts'] = df_.apply(associate_macro_concepts, axis=1, concept_to_macro=concept_to_macro)

    # Ajouter des colonnes dummy pour chaque macro-concept
    unique_macro_concepts = set(concept for concepts in df_['macro_concepts'] for concept in concepts)
    new_columns = {f'dummy_{macro_concept}': df_['macro_concepts'].apply(lambda x: 1 if macro_concept in x else 0) for macro_concept in unique_macro_concepts}
    df_ = pd.concat([df_, pd.DataFrame(new_columns)], axis=1)

    # Sauvegarder le DataFrame final
    df_.to_csv(f"{save_path}/df_with_topics_v4_{dataset}.csv", index=False)
    print(f"Results saved to {save_path}/df_with_topics_v4_{dataset}.csv")

#------------------------------- DATASET/DATA LOADER AUGMENTED (with macro concepts columns) CREATION ----------------

import torch
from torch.utils.data import Dataset as Dataset2
from torch.utils.data import DataLoader

def clean_concept_name(name):
    """
    Nettoie un nom de concept en supprimant le préfixe "cos_" ou "dummy_", en retirant les espaces superflus
    et en supprimant les phrases inutiles à partir de "Let me know...".
    """
    name = name.replace("cos_", "").replace("dummy_", "").strip()
    name = re.sub(r"Let me know.*", "", name)
    return " ".join(name.split())
    
class CustomDataset(Dataset2):
    def __init__(self, dataframe, tokenizer, max_len):
        self.dataframe = dataframe
        self.texts = dataframe['text'].tolist()
        self.labels = dataframe['label'].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # Filtre pour obtenir toutes les colonnes "dummy_"
        self.additional_columns = [col for col in dataframe.columns if col.startswith("dummy_")]
        self.additional_features_keys = [clean_concept_name(col) for col in self.additional_columns]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Création du dictionnaire des additional features (les concepts)
        additional_features = {
            key: torch.tensor(self.dataframe.iloc[idx][dummy_col], dtype=torch.long)
            for key, dummy_col in zip(self.additional_features_keys, self.additional_columns)
        }

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt"
        )

        # Fusionner les dictionnaires
        final_dic = {
            **additional_features,  # Ajoute les additional features
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

        return final_dic

def create_dataloader(df_filtre, tokenizer, max_len, batch_size, shuffle = True):
    # Créer une instance du dataset avec les bons paramètres
    dataset = CustomDataset(df_filtre, tokenizer, max_len)
    # Utiliser cette instance dans le DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader





