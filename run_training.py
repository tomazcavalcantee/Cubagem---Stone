import os
import re
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime

from src.config  import CATEGORIES, CSV_FILE, BATCH_SIZE, NUM_EPOCHS, PATIENCE, LR, SEED, MODELS_DIR, RESULTS_DIR
from src.dataset import get_dataloaders
from src.model   import build_model
from src.train   import train

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

os.makedirs(MODELS_DIR, exist_ok=True)
existing = [f for f in os.listdir(MODELS_DIR) if re.match(r'v(\d+)\.pth$', f)]
next_version = max((int(re.match(r'v(\d+)\.pth$', f).group(1)) for f in existing), default=0) + 1
model_path = os.path.join(MODELS_DIR, f"v{next_version}.pth")
print(f"Modelo será salvo em: {model_path}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando o device: {device}")

train_loader, val_loader = get_dataloaders(CSV_FILE, CATEGORIES, BATCH_SIZE)
model = build_model(device)
history = train(model, train_loader, val_loader, device, NUM_EPOCHS, PATIENCE, LR, model_path)

n = len(history['train_loss'])
epochs = np.arange(1, n + 1)

train_loss  = np.array(history['train_loss'])
train_std   = np.array(history['train_loss_std'])
val_loss    = np.array(history['val_loss'])
val_std     = np.array(history['val_loss_std'])
val_mape     = np.array(history['val_mape'])
val_mape_std = np.array(history['val_mape_std'])

version_tag = f"v{next_version}"
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(epochs, train_loss, 'o-', label='Train Loss')
ax1.fill_between(epochs, train_loss - train_std, train_loss + train_std, alpha=0.2)
ax1.plot(epochs, val_loss, 'o-', label='Val Loss')
ax1.fill_between(epochs, val_loss - val_std, val_loss + val_std, alpha=0.2)
ax1.set_title('Train Loss vs Val Loss (± std)')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('SmoothL1 Loss (log-space)')
ax1.legend()
ax1.grid(True)

ax2.plot(epochs, val_mape, 'o-', color='tomato', label='Val MAPE')
ax2.fill_between(epochs, val_mape - val_mape_std, val_mape + val_mape_std, alpha=0.2, color='tomato')
ax2.set_title('MAPE na Validação (± std)')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('MAPE (%)')
ax2.legend()
ax2.grid(True)

os.makedirs(RESULTS_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = os.path.join(RESULTS_DIR, f"run_{timestamp}_{version_tag}.png")

plt.tight_layout()
plt.savefig(output_path, dpi=150)
print(f"Gráfico salvo em {output_path}")
