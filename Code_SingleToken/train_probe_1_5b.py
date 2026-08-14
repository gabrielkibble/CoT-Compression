"""
Adapted from Bowen1911/Difficulty-Perception-of-LLMs' README training-loop
code block, for DeepSeek-R1-Distill-Qwen-1.5B.

Requires data/deepmath_embedding_1.5b.parquet from extract_embeddings_1_5b.py
to already exist.

Run inside your CREST container:
    python3 train_probe_1_5b.py
"""
import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

EMBEDDING_PATH = "data/deepmath_embedding_1.5b.parquet"
PROBE_OUTPUT_PATH = "models/difficulty_probe_deepseek_r1_1.5b.pth"
HIDDEN_SIZE = 1536  # DeepSeek-R1-Distill-Qwen-1.5B's actual hidden_size (confirmed via AutoConfig)

# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
dfd = pd.read_parquet(EMBEDDING_PATH)

X = np.stack(dfd["emb"].values).astype(np.float32)
y = dfd["difficulty"].values.astype(np.float32).reshape(-1, 1)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_size = int(0.8 * len(X_train_full))
X_train, X_val = torch.from_numpy(X_train_full[:train_size]), torch.from_numpy(X_train_full[train_size:])
y_train, y_val = torch.from_numpy(y_train_full[:train_size]), torch.from_numpy(y_train_full[train_size:])

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class RegressionNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)  # only a linear projection

    def forward(self, x):
        return self.linear(x)

model = RegressionNN(input_dim=HIDDEN_SIZE).to("cuda")

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=2e-4)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
epochs = 80
train_losses, val_losses = [], []

for epoch in range(epochs):
    model.train()
    epoch_train_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to("cuda"), yb.to("cuda")
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item() * xb.size(0)
    epoch_train_loss /= len(train_loader.dataset)
    train_losses.append(epoch_train_loss)

    model.eval()
    epoch_val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to("cuda"), yb.to("cuda")
            pred = model(xb)
            loss = criterion(pred, yb)
            epoch_val_loss += loss.item() * xb.size(0)
    epoch_val_loss /= len(val_loader.dataset)
    val_losses.append(epoch_val_loss)

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

# ---------------------------------------------------------------------------
# Save probe + report test-set performance
# ---------------------------------------------------------------------------
os.makedirs(os.path.dirname(PROBE_OUTPUT_PATH), exist_ok=True)
torch.save(model, PROBE_OUTPUT_PATH)
print(f"Saved probe to {PROBE_OUTPUT_PATH}")

model.eval()
with torch.no_grad():
    X_test_t = torch.from_numpy(X_test).to("cuda")
    y_test_t = torch.from_numpy(y_test).to("cuda")
    test_pred = model(X_test_t)
    test_loss = criterion(test_pred, y_test_t).item()
print(f"Test MSE: {test_loss:.4f}")
