import pandas as pd
from neural import *
from neural.model.model import CubagemMLP

if __name__ == "__main__":
    df = pd.read_csv("data/raw/cubagem_40k_amazon.csv").set_index("asin", drop=True)


    train_mask      = df["split"] == "train"
    train_asin      = list(df[train_mask].index)
    test_asin       = list(df[df["split"] == "val"].index)
    df_train        = df[train_mask].reset_index(drop=True)

    # Tabular features
    feature_processor = build_feature_processor(
        FEATURES["numerical"],
        FEATURES["title_regex"],
        FEATURES["categorical"]
    )
    feature_processor.fit_transform(df_train)
    tabular_features = dict(zip(
        df.index,
        feature_processor.transform(df)
    ))
    
    # Image and Text
    image_embeddings = load_embeddings("embeddings/image_embeddings.npz")
    text_embeddings = load_embeddings("embeddings/text_embeddings.npz")

    # Targets
    c = 5
    base = 1.1
    log_transform = LogTransform(f"log_{base}(x+{c})", c, base)
    log_targets = log_transform.transform(df[TARGETS])
   
    print(len(train_asin), len(test_asin))

    train_dataloader = create_loader(
        image_embeddings, text_embeddings,
        tabular_features, log_targets,
        train_asin
    )

    test_dataloader = create_loader(
        image_embeddings, text_embeddings,
        tabular_features, log_targets,
        test_asin
    )


    dims = dict(
        emb_texto_dim=768,
        emb_img_dim=768,
        feats_dim=len(list(tabular_features.values())[0])
    )

    model = CubagemMLP(**dims)

    trained_model, history = train(
        model, dl_train, dl_val, n_epochs=100, batch_size=16, lr=1e-3,
    )


    #
    # dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE,
    #                    shuffle=True,  num_workers=2, pin_memory=True)
    # dl_vl = DataLoader(ds_vl, batch_size=BATCH_SIZE,
    #                    shuffle=False, num_workers=2, pin_memory=True)
