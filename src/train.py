import os
import logging
import random
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import torch.nn as nn
from matplotlib import pyplot as plt
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from src.workout_dataset import WorkoutDataset
from src.model import NeuralNetwork
from src.rmsle_loss import RMSLELoss


logger: logging.Logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Seed set to {seed}")


def train(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: nn.Module,
    epochs: int = 100,
    lr: float = 1e-3,
    momentum: float = 0.9
) -> None:
    # Funkcja kosztu
    criterion: nn.Module = RMSLELoss()

    # Algorytm optymalizacyjny
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    best_rmsle: float = float("inf")
    train_losses: list[float] = []
    val_losses: list[float] = []

    # Trening przez zdefiniowaną liczbę epok
    for epoch in range(epochs):
        model.train()
        running_loss: float = 0.0
        # Iteracja po wszystkich batchach w zbiorze treninigowym
        for X_batch, y_batch in train_loader:
            # Wyzeruj wszystkie gradienty, w przeciwnym razie stare gradienty będą się akumulować
            optimizer.zero_grad()

            # Predykcja modelu
            logits = model(X_batch)

            # Obliczenie funkcji straty
            loss = criterion(logits, y_batch)
            running_loss += loss.item()

            # Obliczenie gradientów
            loss.backward()

            # Aktualizacja wag na podstawie obliczonych gradientów
            optimizer.step()
        
        running_loss /= len(train_loader)
        train_losses.append(running_loss)

        if epoch % 10 == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                # Ewaluacja na zbiorze walidacyjnym
                for X_val, y_val in val_loader:
                    pred = model(X_val)
                    val_loss += criterion(pred, y_val).item()
                
                # Uśrednienie błędu (zależy czy liczymy średnią per batch czy per sample, 
                # tutaj uproszczenie: suma średnich z batchy / liczba batchy)
                val_loss /= len(val_loader)
                
                logger.info(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {running_loss:.4f} - Val RMSLE: {val_loss:.4f}")
                
                if val_loss < best_rmsle:
                    best_rmsle = val_loss
                    save_path: str = os.path.join(HydraConfig.get().run.dir, "best_model.pth")
                    torch.save(model.state_dict(), save_path)
    
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg : DictConfig) -> None:
    # Ustawienie seeda
    if "seed" in cfg["training"]:
        set_seed(cfg["training"]["seed"])

    # Przygotowanie datasetu
    full_dataset: Dataset = WorkoutDataset(cfg["data"]["train"])
    
    # Podział na treningowy i walidacyjny
    val_split = cfg["training"].get("val_split", 0.2)
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg["training"].get("seed", 42))
    )
    
    logger.info(f"Train size: {train_size}, Val size: {val_size}")

    train_loader: DataLoader = DataLoader(train_dataset, batch_size=cfg["training"]["batch_size"], shuffle=True)
    val_loader: DataLoader = DataLoader(val_dataset, batch_size=cfg["training"]["batch_size"], shuffle=False)

    # Zdefiniowanie modelu
    model: nn.Module = NeuralNetwork()

    # Trenowanie modelu
    train(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        epochs=cfg["training"]["epochs"],
        lr=cfg["training"]["learning_rate"],
        momentum=cfg["training"]["momentum"]
    )


if __name__ == "__main__":
    main()
