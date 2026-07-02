import os
import json
import warnings
import torch
import requests

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from io import BytesIO
from tqdm import tqdm

def load_embeddings(path):
    data = np.load(path, allow_pickle=True)
    return dict(zip(data["asins"], data["embeddings"]))


