import numpy as np
import torch
import torch.nn as nn


def to_tensor(array, dtype=torch.float32):
    return torch.tensor(np.asarray(array, dtype=np.float32), dtype=dtype)


def train_classifier(model, X_train, y_train, X_val=None, y_val=None,
                      epochs: int = 30, lr: float = 0.001, batch_size: int = 64,
                      print_every: int = 5, random_state: int = 42):
    torch.manual_seed(random_state)

    X_train_t = to_tensor(X_train)
    y_train_t = to_tensor(y_train).reshape(-1, 1)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_samples = X_train_t.shape[0]
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(n_samples)
        epoch_loss = 0.0

        for start in range(0, n_samples, batch_size):
            idx = permutation[start:start + batch_size]
            batch_x = X_train_t[idx]
            batch_y = y_train_t[idx]

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_x.size(0)

        epoch_loss /= n_samples
        history["train_loss"].append(epoch_loss)

        val_loss = None
        if X_val is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                X_val_t = to_tensor(X_val)
                y_val_t = to_tensor(y_val).reshape(-1, 1)
                val_logits = model(X_val_t)
                val_loss = criterion(val_logits, y_val_t).item()
            history["val_loss"].append(val_loss)

        if epoch % print_every == 0 or epoch == epochs - 1:
            msg = f"Epoch {epoch:3d} | train_loss: {epoch_loss:.4f}"
            if val_loss is not None:
                msg += f" | val_loss: {val_loss:.4f}"
            print(msg)

    return model, history


def predict_proba(model, X):
    model.eval()
    with torch.no_grad():
        X_t = to_tensor(X)
        logits = model(X_t)
        probs = torch.sigmoid(logits).numpy().flatten()
    return probs