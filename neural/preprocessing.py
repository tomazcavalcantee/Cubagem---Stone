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
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

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

