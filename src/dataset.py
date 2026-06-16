import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class AmazonDimensionsDataset(Dataset):
    def __init__(self, csv_file, split='train', transform=None, categorias=[]):
        self.df = pd.read_csv(csv_file)

        if categorias:
            self.df = self.df[self.df['categories'].isin(categorias)]

        if 'split' in self.df.columns:
            self.df = self.df[self.df['split'] == split]

        self.transform = transform
        self.df = self.df.dropna(subset=['image_url', 'length_cm', 'width_cm', 'height_cm'])
        self.df = self.df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            response = requests.get(row['image_url'], timeout=5)
            img = Image.open(BytesIO(response.content)).convert('RGB')
        except Exception:
            return self.__getitem__((idx + 1) % len(self))

        if self.transform:
            img = self.transform(img)

        dims = sorted([row['length_cm'], row['width_cm'], row['height_cm']], reverse=True)
        return img, torch.tensor(dims, dtype=torch.float32)


def get_dataloaders(csv_file, categorias, batch_size):
    transform = get_transform()
    train_ds = AmazonDimensionsDataset(csv_file, split='train', transform=transform, categorias=categorias)
    val_ds   = AmazonDimensionsDataset(csv_file, split='val',   transform=transform, categorias=categorias)

    print(f"Train: {len(train_ds)} amostras | Val: {len(val_ds)} amostras")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader
