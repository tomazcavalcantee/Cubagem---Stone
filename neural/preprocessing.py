import os, warnings, joblib
import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer, LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

features = {
    "categorical": [
        "main_category",
        "source_category",
    ],
    "high_cardinality_categorical": [
        "categories",
        "store",
    ],
    "text": [
        "title",
        "description",
    ],
    "numerical": [
        "price_numeric",
        "avg_rating",
        "n_ratings",
        "n_images",
        "title_length",
        "title_word_count",
        "description_length",
        "n_categories_levels",
    ],
    "title_regex": [
        "has_dim_in_title",
        "has_size_word_in_title",
        "has_qty_in_title",
    ],
    "image": [
        "image_url",
    ],
}

targets = [
    "length_cm",   # length >= width >= height
    "width_cm",
    "height_cm",
    "weight_g",
]

log_targets = [
    "log_length_cm",   # length >= width >= height
    "log_width_cm",
    "log_height_cm",
    "log_weight_g",
]

class LogTransform:
    def __init__(self, c: float, base: float = np.e, name: str):
        self.c = c
        self.log_base = np.log(base)
        self.name = name

    def __str__(self):
        return self.name

    def transform(self, x):
        return np.log(x + self.c) / self.log_base

    def inverse_transform(self, x):
        return np.exp(x * self.log_base) - self.c


class OneHotTransform:
    encoder: OneHotEncoder
    def __init__(self, cols: list[str], drop_first: bool = False):
        self.encoder = OneHotEncoder(
            sparse_output=False,
            drop="first" if drop_first else None,
            handle_unknown="ignore",
        )
        self.cols = cols

    def fit_transform(self, df: pd.DataFrame):
        encoded = self.encoder.fit_transform(df[self.cols])
        encoded_df = pd.DataFrame(
            encoded,
            columns = self.encoder.get_feature_names_out(self.cols),
            index = df.index,
        )
     
        return df
            .drop(columns=self.cols)
            .join(encoded_df)

    def transform(self, df: pd.DataFrame):
        encoded = self.encoder.transform(df[self.cols])
        encoded_df = pd.DataFrame(
            encoded,
            columns = self.encoder.get_feature_names_out(self.cols),
            index = df.index,
        )
        return df
            .drop(columns=self.cols)
            .join(encoded_df)



