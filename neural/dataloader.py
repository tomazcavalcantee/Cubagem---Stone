import os, warnings, joblib
import torch

import numpy as np
import pandas as pd

def load_embeddings(path):
    data = np.load(path, allow_pickle=True)

    return dict(zip(data["asins"], data["embeddings"]))

class CubagemDataset(Dataset):
    def __init__(self, text_npz_path, img_npz_path, features_dict, targets_dict):
        self.df = df.reset_index(drop=True)
        self.asins = self.df["asin"].values
        
        # Load embeddings
        text_data = np.load(text_npz_path, allow_pickle=True)
        img_data  = np.load(img_npz_path, allow_pickle=True)
        
        self.text_dict = dict(zip(text_data["asins"], text_data["embeddings"]))
        self.img_dict  = dict(zip(img_data["asins"], img_data["embeddings"]))
        
        self.features   = torch.tensor(self.df[features_columns].values, dtype=torch.float32)
        self.targets     = torch.tensor(self.df[targets_columns].values, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        asin = self.asins[idx]
        
        # Fetch embeddings dynamically
        emb_t = self.text_dict.get(asin, np.zeros(768, dtype=np.float32))
        emb_i = self.img_dict.get(asin, np.zeros(768, dtype=np.float32))
        
        return {
            "emb_texto":   torch.tensor(emb_t, dtype=torch.float32),
            "emb_img":     torch.tensor(emb_i, dtype=torch.float32),
            "features":    self.features[idx],
            "targets":     self.targets[idx],
        }
