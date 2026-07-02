import os, warnings, joblib
import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer, LabelEncoder, StandardScaler
from sklearn.impute import IterativeImputer 

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
    def __init__(self, name: str, c: float, base: float = np.e):
        self.c = c
        self.log_base = np.log(base)
        self.name = name

    def __str__(self):
        return self.name

    def transform(self, x):
        return np.log(x + self.c) / self.log_base

    def inverse_transform(self, x):
        return np.exp(x * self.log_base) - self.c

def build_feature_processor(num_cols, regex_cols, cat_cols):
    num_pipeline = Pipeline([
        ("imputer", IterativeImputer(max_iter=10)),
        ("scaler", StandardScaler())
    ])

    regex_pipeline = Pipeline([
        ("convert", FunctionTransformer(lambda x: x.fillna(False).astype(float)))
    ])
     
    cat_pipeline = Pipeline([
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # 4. Stitch them together in parallel
    preprocessor = ColumnTransformer(
        transformers=[
            ("num",     num_pipeline,   num_cols),
            ("regex",   regex_pipeline, regex_cols),
            ("cat",     cat_pipeline,   cat_cols)
        ],
        remainder="drop"  # Safely ignores unlisted columns like 'asin' or 'target'
    )
    
    return preprocessor


