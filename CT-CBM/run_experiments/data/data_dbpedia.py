import torch
import pandas as pd
from models.utils import load_model_and_tokenizer
from torch.utils.data import Dataset, DataLoader
from config_dbpedia import Config
import json
import os

from sklearn.model_selection import train_test_split

def prepare_dbpedia_data(config):
    """
    Prépare et retourne les DataLoader pour le dataset xxx xx.
    
    Cette fonction effectue les opérations suivantes :
      - Détermine le répertoire de sauvegarde en fonction de config.infra.
      - Charge les données prétraitées depuis des fichiers pickle si disponibles,
        sinon effectue le prétraitement (limitation du nombre d'exemples par classe,
        séparation en train/validation/test) et sauvegarde les DataFrame en pickle.
      - Charge le tokenizer via load_model_and_tokenizer.
      - Crée une classe customisée OurCustomDataset.
      - Instancie les datasets et retourne les DataLoader pour l'entraînement,
        la validation et le test.
    
    Paramètres :
      - config : objet de configuration contenant notamment
          - config.max_len (int) : longueur maximale pour le tokenizing
          - config.infra (str) : infrastructure (ex: "DATABRICKS" ou autre)
          - config.batch_size (int) : taille du batch pour le DataLoader
          - ... (d'autres paramètres éventuellement requis par load_model_and_tokenizer)
    
    Retourne :
      - train_loader, val_loader, test_loader, train_df, val_df, test_df : les DataLoader et dataframe respectifs.
    """
    max_len = config.max_len
    infra = config.infra
    
    # Concept-discovery artefacts, relative to the repository root
    save_path_dictionary = "results/dbpedia/concepts_discovery"
    os.makedirs(save_path_dictionary, exist_ok=True)
    
    # Fichiers de sauvegarde en pkl
    train_file = f"{save_path_dictionary}/train_data.pkl"
    val_file = f"{save_path_dictionary}/val_data.pkl"
    test_file = f"{save_path_dictionary}/test_data.pkl"
    
    if os.path.exists(train_file) and os.path.exists(val_file) and os.path.exists(test_file):
        print("Chargement des fichiers sauvegardés...")
        train_df = pd.read_pickle(train_file)
        val_df = pd.read_pickle(val_file)
        test_df = pd.read_pickle(test_file)
    else:
        print("Prétraitement des données...")
    
        # Load data
        splits = {'train': 'dbpedia_14/train-00000-of-00001.parquet', 'test': 'dbpedia_14/test-00000-of-00001.parquet'}
        train_data = pd.read_parquet("hf://datasets/fancyzhx/dbpedia_14/" + splits["train"])
        test_data = pd.read_parquet("hf://datasets/fancyzhx/dbpedia_14/" + splits["test"])
    
        # Rename columns for consistency
        train_data.rename(columns={"content": "text", "label": "label"}, inplace=True)
        test_data.rename(columns={"content": "text", "label": "label"}, inplace=True)
    
        caption_to_number = {
            "Company": 0,
            "Educational Institution": 1,
            "Artist": 2,
            "Athlete": 3,
            "Office Holder": 4,
            "Mean Of Transportation": 5,
            "Building": 6,
            "Natural Place": 7,
            "Village": 8,
            "Animal": 9,
            "Plant": 10,
            "Album": 11,
            "Film": 12,
            "Written Work": 13
        }
    
        # Select only 6 specific classes
        selected_classes = ["Company", "Artist", "Athlete", "Film", "Building", "Animal"]
        selected_labels = {caption_to_number[cls] for cls in selected_classes}
        train_data = train_data[train_data["label"].isin(selected_labels)]
    
        # Re-label the selected classes to 0, 1, 2, 3, 4, 5 for easier computation in PyTorch
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(selected_labels)}
        # {0: 0, 2: 1, 3: 2, 6: 3, 9: 4, 12: 5}
        # so new labelisation is {"Company": 0, "Artist": 1, "Athlete": 2, "Building": 3, "Animal": 4, "Film": 5}
        os.makedirs(save_path_dictionary, exist_ok = True)
        with open(f"{save_path_dictionary}/dictionary_dbpedia.json", 'w') as f:
            json.dump(label_mapping, f)
    
        train_data["label"] = train_data["label"].map(label_mapping)
        test_data = test_data[test_data["label"].isin(selected_labels)]
        test_data["label"] = test_data["label"].map(label_mapping)
    
        # Limit training data to 1000 examples per class
        train_data_limited = train_data.groupby("label", group_keys=False).apply(lambda x: x.sample(n=1000, random_state=42))
    
        # Use remaining data for validation
        remaining_data = train_data.drop(train_data_limited.index)
    
        # Convert data to pandas DataFrames
        train_df = train_data_limited[["text", "label"]]
        val_df = remaining_data[["text", "label"]]
        test_df = test_data[["text", "label"]]
    
        # save to pickle
        train_df.to_pickle(train_file)
        val_df.to_pickle(val_file)
        test_df.to_pickle(test_file)
        print("Données sauvegardées.")
    
    # Import the tokenizer
    _m, tokenizer, _, _c = load_model_and_tokenizer(config)
    
    class OurCustomDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_len = max_len
    
        def __len__(self):
            return len(self.texts)
    
        def __getitem__(self, idx):
            text = self.texts[idx]
            label = self.labels[idx]
    
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                truncation=True,
                padding="max_length",
                return_attention_mask=True,
                return_tensors="pt"
            )
    
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'label': torch.tensor(label, dtype=torch.long)
            }
    
    
    # Create dataset instances
    train_dataset = OurCustomDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer, max_len)
    val_dataset = OurCustomDataset(val_df["text"].tolist(), val_df["label"].tolist(), tokenizer, max_len)
    test_dataset = OurCustomDataset(test_df["text"].tolist(), test_df["label"].tolist(), tokenizer, max_len)
    
    # Create DataLoader instances
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    return train_loader, test_loader, val_loader, train_df, val_df, test_df 