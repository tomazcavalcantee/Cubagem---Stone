import os
import torch
import requests

import numpy as np
import pandas as pd

from PIL import Image
from io import BytesIO
from tqdm import tqdm
from embeddings import load_embeddings
from transformers import AutoModel, AutoImageProcessor
from concurrent.futures import ThreadPoolExecutor, as_completed

def download_image(url, asin, path="data/images"):
    """Baixa a imagem de um produto e salva em disco. Retorna (asin, ok)."""
    path = os.path.join(path, f"{asin}.jpg")
    if os.path.exists(path):
        return asin, True
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        img.save(path, "JPEG")
        return asin, True
    except Exception:
        return asin, False

def parallel_image_download(url_dict, n_workers=8):
    falhas = []
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(download_image, url, asin): url for asin, url in url_dict.items()}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Download"):
            asin, ok = fut.result()
            if not ok:
                falhas.append(asin)
    print(f"Failed Downloads: {len(falhas)}")
    return set(falhas)

def load_image(asin, image_path="data/images"):
    path = os.path.join(image_path, f"{asin}.jpg")
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return Image.new("RGB", (224, 224), color=128)

def extract_embeddings(url_dict, processor, model, embeddings_path="embeddings/imagem_embeddings.npz"):
    if os.path.exists(embeddings_path):
        print(f"Embeddings exist. Loading.")
        return load_embeddings(embeddings_path)

    asins = list(url_dict.keys())
    embeddings  = []
    model.eval()
    for i in tqdm(range(0, len(asins), 32), desc=f"DINOv2"):
        batch_asins = asins[i : i + 32]
        imgs = [load_image(a) for a in batch_asins]
        enc = processor(images=imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            out  = dino_model(**enc)
            # Get image global embedding 
            pool = out.last_hidden_state[:, 0, :]
            embs.append(pool.cpu().numpy())
    np_embeddings = np.concatenate(embeddings, axis=0)

    np.savez(
        embeddings_path,
        embeddings=np_embeddings,
        asins=np.asarray(asins)
    )
    return dict(zip(asins, np_embeddings))

if __name__ == "__main__":
    df = pd.read_csv("data/raw/cubagem_40k_amazon.csv").set_index("asin", drop=True)
    url_dict = df["image_url"].to_dict()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    dino_model     = AutoModel.from_pretrained("facebook/dinov2-base").to(device)

    for p in dino_model.parameters():
        p.requires_grad = False

    extract_embeddings(url_dict, dino_processor, dino_model)

    del dino_model
    torch.cuda.empty_cache()
