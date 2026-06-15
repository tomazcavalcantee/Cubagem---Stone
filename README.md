# Cubagem Stone

## Running the experiments
- Install [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager.
- In the terminal, run `uv sync`
- Run `uv run jupyter notebook` to start the Jupyter environment 

## `data/` 
- `cubagem_40k_amazon.csv` — dataset completo (40000 linhas)
- `cubagem_40k_amazon_train.csv` — train (32000 linhas)
- `cubagem_40k_amazon_val.csv` — validation (8000 linhas)

## Objectives 
- `length_cm`, `width_cm`, `height_cm` — dimensoes (length >= width >= height)
- `weight_g` — peso
- `volume_cm3` — derivado, mas tambem pode ser alvo direto

## Features 
- Categóricas: `main_category`, `source_category`, `categories` (hierarquia), `store`
- Texto: `title`, `description`
- Numericas auxiliares: `price_numeric`, `avg_rating`, `n_ratings`, `n_images`,
  `title_length`, `title_word_count`, `description_length`, `n_categories_levels`
- Sinais regex do titulo: `has_dim_in_title`, `has_size_word_in_title`, `has_qty_in_title`
- Visao computacional: `image_url` (url da imagem grande)

## Splits
- 80/20 train/val, estratificado por `main_category`, seed=42.
- Coluna `split` indica em qual particao cada linha esta.

## Filtros aplicados
- Dimensoes em [0.5, 200.0] cm
- Peso em [1.0, 50000.0] g
- Volume >= 1.0 cm3
- Densidade em [0.005, 5.0] g/cm3
- Titulo com >= 5 caracteres
- Dedup por ASIN
