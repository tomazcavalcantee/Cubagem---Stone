import torch

import numpy as np

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
