"""
Deep Learning architectures for LOB prediction.

Implements:
- CNN: 1D convolutions across order book levels (spatial patterns)
- LSTM: Sequential temporal modeling
- CNN-LSTM hybrid: spatial feature extraction + temporal modeling

Inspired by DeepLOB (Zhang et al., 2019) and recent LOB forecasting literature.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import structlog

logger = structlog.get_logger(__name__)


# ─── Dataset ──────────────────────────────────────────────────────────────────

class LOBDataset(Dataset):
    """
    PyTorch Dataset for LOB feature sequences.

    Each sample is a sequence of T consecutive feature vectors → 1 label.
    The label corresponds to the LAST timestep in the sequence.
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        seq_length: int = 50,
    ) -> None:
        """
        Args:
            features: Array of shape (N, n_features)
            labels: Array of shape (N,) with integer labels
            seq_length: Number of timesteps per sequence
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.seq_length = seq_length

    def __len__(self) -> int:
        return len(self.features) - self.seq_length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features[idx : idx + self.seq_length]  # (seq_length, n_features)
        y = self.labels[idx + self.seq_length - 1]       # Label at end of sequence
        return x, y


# ─── Models ───────────────────────────────────────────────────────────────────

class LOB_CNN(nn.Module):
    """
    1D CNN for order book pattern recognition.

    Architecture:
    - Conv1D layers extract spatial patterns across price levels
    - BatchNorm + Dropout for regularization
    - Fully connected classifier head

    Input shape: (batch, seq_length, n_features)
    Output shape: (batch, 3) — probabilities for DOWN/NEUTRAL/UP
    """

    def __init__(
        self,
        n_features: int,
        seq_length: int = 50,
        n_classes: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.conv_block = nn.Sequential(
            # Block 1
            nn.Conv1d(n_features, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),

            # Block 2
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),

            # Block 3
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_length, n_features) → transpose to (batch, n_features, seq_length)
        x = x.permute(0, 2, 1)
        x = self.conv_block(x)       # (batch, 64, 1)
        x = x.squeeze(-1)            # (batch, 64)
        x = self.classifier(x)       # (batch, n_classes)
        return x


class LOB_LSTM(nn.Module):
    """
    LSTM for temporal sequence modeling of LOB features.

    Architecture:
    - Multi-layer LSTM
    - Fully connected classifier on final hidden state

    Input shape: (batch, seq_length, n_features)
    Output shape: (batch, 3)
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        n_layers: int = 2,
        n_classes: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=False,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_length, n_features)
        lstm_out, (h_n, _) = self.lstm(x)     # h_n: (n_layers, batch, hidden)
        last_hidden = h_n[-1]                  # (batch, hidden)
        return self.classifier(last_hidden)     # (batch, n_classes)


class LOB_CNN_LSTM(nn.Module):
    """
    Hybrid CNN-LSTM for LOB prediction.

    Architecture:
    - CNN extracts local spatial features from each timestep
    - LSTM models the temporal evolution of CNN features
    - FC classifier head

    Input shape: (batch, seq_length, n_features)
    Output shape: (batch, 3)
    """

    def __init__(
        self,
        n_features: int,
        seq_length: int = 50,
        cnn_channels: int = 32,
        lstm_hidden: int = 64,
        n_classes: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        # Per-timestep CNN (applied independently to each timestep)
        self.cnn = nn.Sequential(
            nn.Conv1d(1, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(n_features // 2),
        )

        cnn_out_size = cnn_channels * (n_features // 2)

        self.lstm = nn.LSTM(
            input_size=cnn_out_size,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, n_feat = x.shape

        # Apply CNN to each timestep
        cnn_features = []
        for t in range(seq_len):
            xt = x[:, t, :].unsqueeze(1)     # (batch, 1, n_features)
            ct = self.cnn(xt)                  # (batch, cnn_channels, n_features//2)
            ct = ct.view(batch_size, -1)       # (batch, cnn_channels * n_features//2)
            cnn_features.append(ct)

        cnn_seq = torch.stack(cnn_features, dim=1)  # (batch, seq_len, cnn_out_size)

        # LSTM on CNN features
        _, (h_n, _) = self.lstm(cnn_seq)
        last_hidden = h_n[-1]

        return self.classifier(last_hidden)


# ─── Training ─────────────────────────────────────────────────────────────────

def train_dl_model(
    model: nn.Module,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray | None = None,
    val_labels: np.ndarray | None = None,
    seq_length: int = 50,
    batch_size: int = 256,
    n_epochs: int = 50,
    learning_rate: float = 1e-3,
    patience: int = 10,
    device: str = "auto",
) -> dict[str, list[float]]:
    """
    Train a deep learning model with early stopping.

    Args:
        model: PyTorch model
        train_features: Training features (N, n_features)
        train_labels: Training labels (N,)
        val_features: Validation features (optional)
        val_labels: Validation labels (optional)
        seq_length: Sequence length for dataset
        batch_size: Training batch size
        n_epochs: Maximum epochs
        learning_rate: Learning rate
        patience: Early stopping patience
        device: "auto", "cuda", or "cpu"

    Returns:
        Dict with training history (train_loss, val_loss per epoch)
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    logger.info("dl_training_start", device=device, model=model.__class__.__name__)

    train_ds = LOBDataset(train_features, train_labels, seq_length)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)

    val_loader = None
    if val_features is not None and val_labels is not None:
        val_ds = LOBDataset(val_features, val_labels, seq_length)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=patience // 2, factor=0.5
    )
    criterion = nn.CrossEntropyLoss()

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(n_epochs):
        # Training
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        history["train_loss"].append(avg_train_loss)

        # Validation
        if val_loader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    logits = model(X_batch)
                    loss = criterion(logits, y_batch)
                    val_losses.append(loss.item())

            avg_val_loss = np.mean(val_losses)
            history["val_loss"].append(avg_val_loss)

            scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 5 == 0:
                logger.info(
                    "dl_epoch",
                    epoch=epoch + 1,
                    train_loss=round(avg_train_loss, 4),
                    val_loss=round(avg_val_loss, 4),
                    best_val=round(best_val_loss, 4),
                    lr=optimizer.param_groups[0]["lr"],
                )

            if patience_counter >= patience:
                logger.info("early_stopping", epoch=epoch + 1, best_val_loss=best_val_loss)
                break
        else:
            if (epoch + 1) % 5 == 0:
                logger.info("dl_epoch", epoch=epoch + 1, train_loss=round(avg_train_loss, 4))

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return history


def predict_dl_model(
    model: nn.Module,
    features: np.ndarray,
    seq_length: int = 50,
    batch_size: int = 512,
    device: str = "auto",
) -> np.ndarray:
    """
    Generate predictions from a trained DL model.

    Returns probability array of shape (N - seq_length, 3).
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    # Create sequences
    dummy_labels = np.zeros(len(features), dtype=int)
    ds = LOBDataset(features, dummy_labels, seq_length)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_probs = []
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())

    return np.vstack(all_probs)
