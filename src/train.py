import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def train(model, train_loader, val_loader, device, num_epochs, patience, lr, model_path):
    criterion = nn.SmoothL1Loss().to(device)
    optimizer = optim.Adam(model.fc.parameters(), lr=lr)

    history = {
        'train_loss': [], 'train_loss_std': [],
        'val_loss':   [], 'val_loss_std':   [],
        'val_mae':    [], 'val_mae_std':    [],
    }
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        # --- TREINAMENTO ---
        model.train()
        batch_train_losses = []

        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for inputs, targets in train_bar:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            batch_train_losses.append(loss.item())
            train_bar.set_postfix({'loss': loss.item()})

        epoch_train_loss = np.mean(batch_train_losses)
        epoch_train_std  = np.std(batch_train_losses)

        # --- VALIDAÇÃO ---
        model.eval()
        batch_val_losses = []
        batch_val_maes   = []
        sample_preds     = None
        sample_targets   = None

        val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(val_bar):
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                mae  = torch.mean(torch.abs(outputs - targets)).item()

                batch_val_losses.append(loss.item())
                batch_val_maes.append(mae)
                val_bar.set_postfix({'loss': loss.item(), 'mae': f'{mae:.2f}cm'})

                if i == 0:
                    sample_preds   = outputs.cpu().numpy()
                    sample_targets = targets.cpu().numpy()

        epoch_val_loss    = np.mean(batch_val_losses)
        epoch_val_std     = np.std(batch_val_losses)
        epoch_val_mae     = np.mean(batch_val_maes)
        epoch_val_mae_std = np.std(batch_val_maes)

        history['train_loss'].append(epoch_train_loss)
        history['train_loss_std'].append(epoch_train_std)
        history['val_loss'].append(epoch_val_loss)
        history['val_loss_std'].append(epoch_val_std)
        history['val_mae'].append(epoch_val_mae)
        history['val_mae_std'].append(epoch_val_mae_std)

        print(f'\nEpoch {epoch+1} | '
              f'Train Loss: {epoch_train_loss:.4f} ± {epoch_train_std:.4f} | '
              f'Val Loss: {epoch_val_loss:.4f} ± {epoch_val_std:.4f} | '
              f'Val MAE: {epoch_val_mae:.2f} ± {epoch_val_mae_std:.2f} cm')

        print("-" * 65)
        print(f"{'Previsto (Max, Med, Min) em cm':<32} | {'Real (Max, Med, Min) em cm'}")
        print("-" * 65)
        for p, t in zip(sample_preds[:5], sample_targets[:5]):
            print(f"{[f'{v:.1f}' for v in p]} | {[f'{v:.1f}' for v in t]}")
        print()

        # --- EARLY STOPPING ---
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), model_path)
        else:
            epochs_without_improvement += 1
            print(f'Early stopping: {epochs_without_improvement}/{patience} sem melhora.')
            if epochs_without_improvement >= patience:
                print(f'Parando na epoch {epoch+1}. Melhor val loss: {best_val_loss:.4f}')
                model.load_state_dict(torch.load(model_path))
                break

    return history
