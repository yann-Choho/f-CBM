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

from path_info import PATH
PATH_DATA_N24 = PATH+"/datasets/N24News"

technology_news_concepts = [
    # Subject Matter
    "digital products and services",
    "smartphones",
    "software applications",
    "computing hardware",
    "networking infrastructure",
    "internet platforms",
    "social media",
    "artificial intelligence",
    "machine learning",
    "biotechnology",
    "healthcare technology",
    "robotics",
    "automation systems",
    "cybersecurity",
    "data privacy",
    "cryptocurrency",
    "space technology",
    "clean energy tech",
    
    # Key Organizations
    "tech corporations",
    "startups",
    "venture capital",
    "research institutions",
    "regulatory bodies",
    
    # Technical Terminology
    "APIs",
    "cloud computing",
    "Internet of Things",
    "technical specifications",
    "programming frameworks",
    "technical standards",
    "data analytics",
    "virtual reality",
    "augmented reality",
    "quantum computing",
    
    # Content Structure
    "product reviews",
    "technical explanations",
    "societal impact analysis",
    "industry trends",
    "scientific research applications",
    
    # Common Article Frames
    "innovation narratives",
    "problem-solution discussions",
    "future predictions",
    "competitive market analysis",
    "consumer adoption patterns",
    "ethical considerations",
    "regulatory developments"
]

sports_news_concepts = [
    # Subject Matter
    "team sports",
    "individual athletics",
    "championships",
    "tournaments",
    "leagues",
    "athletic performance",
    "player transfers",
    "rankings",
    "equipment innovations",
    "training methodologies",
    "sports medicine",
    "injuries",
    "Olympics",
    "extreme sports",
    "esports",
    "analytics",
    "management decisions",
    "coaching strategies",
    
    # Key Organizations
    "athletic associations",
    "sports federations",
    "collegiate programs",
    "sporting goods manufacturers",
    "media networks",
    "sponsorship partners",
    
    # Technical Terminology
    "statistics",
    "performance metrics",
    "conditioning techniques",
    "tactical systems",
    "rulebook interpretations",
    "officiating",
    "draft selections",
    "salary regulations",
    "biomechanics",
    "wearable technology",
    
    # Content Structure
    "match reports",
    "athlete profiles",
    "season previews",
    "post-game analysis",
    "commentary",
    
    # Common Article Frames
    "victory narratives",
    "underdog stories",
    "comeback scenarios",
    "rivalry histories",
    "fan culture",
    "historical comparisons",
    "career milestones",
    "controversies",
    "business developments",
    "event preparation"
]

food_news_concepts = [
    # Subject Matter
    "cuisine trends",
    "restaurant developments",
    "chef profiles",
    "cooking techniques",
    "ingredients",
    "recipes",
    "food festivals",
    "dietary patterns",
    "agricultural production",
    "sustainability practices",
    "culinary awards",
    "food safety",
    "nutrition research",
    "beverage industry",
    "specialty diets",
    "food technology",
    "cultural traditions",
    "market pricing",
    
    # Key Organizations
    "restaurants",
    "culinary institutes",
    "food manufacturers",
    "grocery retailers",
    "farmers associations",
    "regulatory agencies",
    "delivery services",
    
    # Technical Terminology
    "flavor profiles",
    "nutritional composition",
    "preparation methods",
    "preservation processes",
    "fermentation",
    "food science",
    "supply chain",
    "seasonal availability",
    "terroir",
    "gastronomic techniques",
    
    # Content Structure
    "restaurant reviews",
    "recipe guides",
    "chef interviews",
    "food origins",
    "ingredient features",
    "tasting evaluations",
    
    # Common Article Frames
    "taste experiences",
    "health connections",
    "cultural significance",
    "environmental impact",
    "innovation stories",
    "heritage perspectives",
    "budget considerations",
    "seasonal focus",
    "ethical sourcing",
    "global influences"
]

music_news_concepts = [
    # Subject Matter
    "album releases",
    "single debuts",
    "concert tours",
    "music festivals",
    "artist collaborations",
    "industry awards",
    "genre evolution",
    "band formations/breakups",
    "record contracts",
    "streaming metrics",
    "instrument innovations",
    "production techniques",
    "music education",
    "composition credits",
    "chart rankings",
    "visual media",
    "soundtrack releases",
    "copyright disputes",
    
    # Key Organizations
    "record labels",
    "streaming platforms",
    "performance venues",
    "music publishers",
    "artist management",
    "distribution networks",
    "rights organizations",
    
    # Technical Terminology
    "musical composition",
    "audio engineering",
    "mastering processes",
    "vocal techniques",
    "harmonic structures",
    "arrangement styles",
    "music theory",
    "production software",
    "acoustics",
    "sound design",
    
    # Content Structure
    "album reviews",
    "artist interviews",
    "performance coverage",
    "lyrical analysis",
    "catalog retrospectives",
    "historical context",
    
    # Common Article Frames
    "artistic development",
    "cultural significance",
    "generational influences",
    "commercial performance",
    "creative methodology",
    "audience response",
    "identity expression",
    "historical importance",
    "technological integration",
    "industry direction"
]

concepts_N24 = {'Technology': technology_news_concepts, 'Sports': sports_news_concepts, 'Food': food_news_concepts, 'Music': music_news_concepts}

class N24NewsDataset(Dataset):
    def __init__(self, root, transform=None, max_len=512, n_instances=None, class_list = None):
        self.root = root
        self.transform = transform
        self.max_len = max_len

        with open(os.path.join(root, 'news/nytimes_dataset.json'), 'r') as file:
            self.dataset = pd.DataFrame(json.load(file))
            if(n_instances is not None):
                self.dataset = self.dataset.sample(n=n_instances)

        self.dataset['image_id'] = self.dataset['image_id'].apply(lambda x: os.path.join(root, 'imgs', x)+'.jpg')
        if(class_list is not None):
            self.dataset = self.dataset[self.dataset['section'].isin(class_list)]

        self.class_dict = {i: c for i, c in enumerate(self.dataset['section'].unique())}

    def __len__(self):
        return len(self.dataset)

    @lru_cache(maxsize=1000)
    def _load_and_transform_image(self, path):
        with Image.open(path) as img:
            if self.transform:
                return self.transform(img)
            return img

    def __getitem__(self, idx):
        image = self._load_and_transform_image(self.dataset.iloc[idx]['image_id'])
        headline = self.dataset.iloc[idx]["headline"][:self.max_len].ljust(self.max_len)
        abstract = self.dataset.iloc[idx]["abstract"][:self.max_len].ljust(self.max_len)
        caption = self.dataset.iloc[idx]["caption"][:self.max_len].ljust(self.max_len)
        text_label = self.dataset.iloc[idx]["section"]
        label = [k for k, v in self.class_dict.items() if v == text_label][0]
        image_id = self.dataset.iloc[idx]['image_id']

        return {"image": image, "headline": headline, "text": abstract, "caption": caption, "label": label, "text_label": text_label, 'image_id': image_id}

class ConvertToRGB:
    def __call__(self, image):
        return image.convert('RGB') if image.mode != 'RGB' else image

def load_data_N24(data_dir, class_list, batch_size=32, num_workers=4, max_len=512):
    transform = transforms.Compose([
                ConvertToRGB(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )])

    full_dataset = N24NewsDataset(root=data_dir, transform=transform, max_len=max_len, class_list=class_list)
    print('Dataset loaded')

    num_classes = len(full_dataset.dataset['section'].unique())
    print(f"Number of classes: {num_classes}")
    print("CHECKPOINT A")

    # create dictionnary to map class index to class names
    class_dict = full_dataset.class_dict

    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    full_labels = full_dataset.dataset['section'].values

    # First split: Train and Test/Validation
    train_indices_raw, test_val_indices_raw = next(skf.split(full_dataset.dataset, full_labels))

    # Second split: Validation and Test (on Test/Validation subset)
    test_val_labels = full_labels[test_val_indices_raw]
    val_indices_raw, test_indices_raw = next(skf.split(full_dataset.dataset.iloc[test_val_indices_raw], test_val_labels))

    # Map raw indices back to original dataset indices
    val_indices = test_val_indices_raw[val_indices_raw]
    test_indices = test_val_indices_raw[test_indices_raw]

    # combine train and test_indices (no test)
    train_indices_raw = np.concatenate((train_indices_raw, test_indices))
    test_indices = []

    # Create subsets
    train_dataset = Subset(full_dataset, train_indices_raw)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)
    print(f"Number of instances - Train: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Print the number of instances in each dataset
    print(f"Number of instances - Train: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = 0#DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, class_dict