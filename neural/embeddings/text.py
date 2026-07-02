import os
import torch

import numpy as np
import pandas as pd

from embeddings import load_embeddings
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

def mean_pool(last_hidden, attention_mask):
    """Pooling pela média dos tokens (ignora padding)."""
    mask = attention_mask.unsqueeze(-1).float()
    return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

def extract_embeddings(text_dict, tokenizer, model, device, path="embeddings/text_embeddings.npz"):
    if os.path.exists(path):
        print(f"Embeddings exist. Loading.")
        return load_embeddings(path)
 
    asins  = list(text_dict.keys())
    texts = list(text_dict.values())
    embeddings   = []
    model.eval()
    for i in tqdm(range(0, len(asins), 64), desc=f"RoBERTa"):
        batch_txt = texts[i : i + 64]
        enc = tokenizer(
            batch_txt,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out  = model(**enc)
            pool = mean_pool(out.last_hidden_state, enc["attention_mask"])
            embeddings.append(pool.cpu().numpy())
    np_embeddings = np.concatenate(embeddings, axis=0)
 
    np.savez(
        path,
        embeddings=np_embeddings,
        asins=asins
    )
 
    return dict(zip(asins, np_embeddings))

if __name__ == "__main__":
    df = pd.read_csv("data/raw/cubagem_40k_amazon.csv").set_index("asin", drop=True)
    df["text"] = df["title"] + df["source_category"] + df["description"]
    df["text"] = df["text"].astype(str).str[:400].fillna("")
    text_dict = df["text"].to_dict()
    print(text_dict)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer  = AutoTokenizer.from_pretrained("hyp1231/blair-roberta-base")
    model      = AutoModel.from_pretrained("hyp1231/blair-roberta-base").to(device)

    for p in model.parameters():
        p.requires_grad = False

    extract_embeddings(text_dict, tokenizer, model, device)

    # free memory 
    del model
    torch.cuda.empty_cache()
