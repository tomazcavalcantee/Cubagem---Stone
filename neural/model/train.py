import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR

class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()

    def forward(self,x,y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y))
        return loss

def move_batch_to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    n_batches = 0
 
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        targets = batch["targets"]
 
        optimizer.zero_grad()
        preds = model(batch)
        loss = loss_fn(preds, targets)
        loss.backward()
        optimizer.step()
 
        total_loss += loss.item()
        n_batches += 1
 
    return total_loss / n_batches

@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
 
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        targets = batch["targets"]
 
        preds = model(batch)
        loss = loss_fn(preds, targets)
  
        total_loss += loss.item()
        n_batches += 1
 
    return total_loss / n_batches

def train(
    model,
    train_loader,
    val_loader,
    n_epochs=50,
    batch_size=64,
    lr=1e-3,
    weight_decay=1e-4,
    device=None,
    patience=3
):
    """
    Training loop.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
  
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ExponentialLR(optimizer, gamma=0.7)
    loss_fn = RMSELoss()
 
    history = []
    best_val_rmse = float("inf")
    best_state_dict = None
 
    for epoch in range(1, n_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = evaluate(model, val_loader, loss_fn, device)
        scheduler.step()
 
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        )
 
        # Track best RMSE and save weights
        if val_loss < best_val_rmse:
            best_val_rmse = val_loss
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
 
        # Print logs
        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d} | "
                f"train loss (RMSE): {train_loss:.4f} | "
                f"val loss (RMSE): {val_loss:.4f} | "
            )

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break
 
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
 
    return model, history
