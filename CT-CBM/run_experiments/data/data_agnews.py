import torch
import pandas as pd
from models.utils import load_model_and_tokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import json
import os

# function because of the tokenizer which depend on the modele : we want to ease the experimentation with this
def prepare_agnews_data(config):
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
    save_path_dictionary = "results/agnews/concepts_discovery"
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
        
        splits = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}
        train_data= pd.read_parquet("hf://datasets/fancyzhx/ag_news/" + splits["train"])
        test_data = pd.read_parquet("hf://datasets/fancyzhx/ag_news/" + splits["test"])
    
        caption_to_number = {'World' : 0, 'Sports' : 1, 'Business' : 2, 'Sci/Tech' : 3}
        os.makedirs(save_path_dictionary, exist_ok = True)
        with open(f"{save_path_dictionary}/dictionary_agnews.json", 'w') as f:
            json.dump(caption_to_number, f)
    
    
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