import torch

from torch.utils.data import TensorDataset, DataLoader

def train_model_torch(model, optimizer, criterion, X_train, y_train, X_test, y_test,
                        epochs=100, batch_size=32, device=None):
    
    #Train the pytorch model
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")

    model.to(device)

    #Convert numpy arrays to torch tensors

    X_train_t = torch.tensor(X_train, dtype = torch.float32)
    y_train_t = torch.tensor(y_train, dtype = torch.float32)
    X_test_t = torch.tensor(X_test, dtype = torch.float32)
    y_test_t = torch.tensor(y_test, dtype = torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation loss
        model.eval()
        with torch.no_grad():
            val_preds = model(X_test_t.to(device))
            val_loss = criterion(val_preds, y_test_t.to(device)).item()

        avg_train_loss = running_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)


        if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{epochs}]  Train Loss: {avg_train_loss:.4f}  Val Loss: {val_loss:.4f}")

    print("Training complete")
    return history


def evaluate_model_torch(model, X_test, y_test, criterion, device=None):

    #Evaluate model on test set.

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        y_pred = model(X_test_t)
        loss = criterion(y_pred, y_test_t).item()

        mae = torch.mean(torch.abs(y_pred - y_test_t)).item()

    print(f"[Evaluation] Test MSE: {loss:.4f}, MAE: {mae:.4f}")
    return loss, mae


