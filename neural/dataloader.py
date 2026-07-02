import os, warnings, joblib
import torch

import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader

from neural.config import *

class CubagemDataset(Dataset):
    def __init__(self, image_embeddings: dict, text_embeddings: dict, tabular_features: dict, targets: dict, subset_keys):
        self.index_dict = {
            i: asin for i, asin 
            in zip(range(len(subset_keys)), subset_keys)
        }

        self.text_embeddings = text_embeddings
        self.image_embeddings = image_embeddings
        self.features = tabular_features
        self.targets = targets

    def __len__(self):
        return len(self.index_dict)

    def __getitem__(self, idx):
        asin = self.index_dict[idx]
        
        # Fetch embeddings dynamically
        emb_t = self.text_dict.get(asin, np.zeros(768, dtype=np.float32))
        emb_i = self.img_dict.get(asin, np.zeros(768, dtype=np.float32))
        
        return {
            "emb_texto":   torch.tensor(emb_t, dtype=torch.float32),
            "emb_img":     torch.tensor(emb_i, dtype=torch.float32),
            "features":    self.features[idx],
            "targets":     self.targets[idx],
        }

def create_loader(image_embeddings, text_embeddings, tabular_features, log_targets, subset_keys):
    dataset = CubagemDataset(
        image_embeddings, text_embeddings,
        tabular_features, log_targets,
        subset_keys 
    )
    
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE,
        shuffle=True,  num_workers=2, pin_memory=True
    )

    return dataloader 

