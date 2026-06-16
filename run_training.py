import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime

from src.config  import CATEGORIES, CSV_FILE, BATCH_SIZE, NUM_EPOCHS, PATIENCE, LR, SEED, MODEL_PATH, RESULTS_DIR
from src.dataset import get_dataloaders
from src.model   import build_model
from src.train   import train

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando o device: {device}")

train_loader, val_loader = get_dataloaders(CSV_FILE, CATEGORIES, BATCH_SIZE)
model = build_model(device)
history = train(model, train_loader, val_loader, device, NUM_EPOCHS, PATIENCE, LR, MODEL_PATH)

n = len(history['train_loss'])
epochs = np.arange(1, n + 1)

train_loss  = np.array(history['train_loss'])
train_std   = np.array(history['train_loss_std'])
val_loss    = np.array(history['val_loss'])
val_std     = np.array(history['val_loss_std'])
val_mae     = np.array(history['val_mae'])
val_mae_std = np.array(history['val_mae_std'])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(epochs, train_loss, 'o-', label='Train Loss')
ax1.fill_between(epochs, train_loss - train_std, train_loss + train_std, alpha=0.2)
ax1.plot(epochs, val_loss, 'o-', label='Val Loss')
ax1.fill_between(epochs, val_loss - val_std, val_loss + val_std, alpha=0.2)
ax1.set_title('Train Loss vs Val Loss (± std)')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('SmoothL1 Loss')
ax1.legend()
ax1.grid(True)

ax2.plot(epochs, val_mae, 'o-', color='tomato', label='Val MAE')
ax2.fill_between(epochs, val_mae - val_mae_std, val_mae + val_mae_std, alpha=0.2, color='tomato')
ax2.set_title('Erro Médio Absoluto (MAE) na Validação (± std)')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('MAE (cm)')
ax2.legend()
ax2.grid(True)

os.makedirs(RESULTS_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = os.path.join(RESULTS_DIR, f"run_{timestamp}.png")

plt.tight_layout()
plt.savefig(output_path, dpi=150)
print(f"Gráfico salvo em {output_path}")
