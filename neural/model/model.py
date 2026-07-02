import torch

import torch.nn as nn

class CubagemMLP(nn.Module):
    """
    Two-layer MLP predicting 4 targets (3 dimensions + weight, in log space)
    from concatenated image/text embeddings + numerical/categorical/regex feats.
    """
    def __init__(
        self,
        emb_texto_dim,
        emb_img_dim,
        feats_dim,
        batch_norm: bool = False,
        hidden_dim=1024,
        n_targets=4,
        dropout=0.5,
    ):
        super().__init__()
 
        input_dim = emb_texto_dim + emb_img_dim + feats_dim

        self.net = nn.Sequential()

        if batch_norm:
            self.net.append(nn.BatchNorm1d(input_dim))
        
        self.net.append([
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim//4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//4, n_targets),
        ])
 
    def forward(self, batch):
        x = torch.cat(
            [
                batch["emb_texto"],
                batch["emb_img"],
                batch["features"],
            ],
            dim=1,
        )
        return self.net(x)
