import os

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import (
    DataLoader,
    TensorDataset
)

# from sensor.sensor_model import SensorAutoencoder
from sensor_model import SensorAutoencoder


# ==========================================
# CONFIGURATION
# ==========================================

DATA_PATH = "/Users/shyamchauhan/Desktop/home/codes/crop_disease_project/dataset/sensor/sensor_processed.csv"

MODEL_PATH = "/Users/shyamchauhan/Desktop/home/codes/crop_disease_project/models/sensor_autoencoder.pth"

BATCH_SIZE = 64

EPOCHS = 100

LEARNING_RATE = 0.001


# ==========================================
# CREATE MODEL DIRECTORY
# ==========================================

os.makedirs("models", exist_ok=True)


# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(DATA_PATH)


FEATURES = [
    "soil_moisture",
    "temperature",
    "humidity",
    "pump_data",
    "temperature_ma",
    "temperature_std",
    "humidity_ma",
    "humidity_std"
]


X = df[FEATURES].values.astype(
    np.float32
)


print("Input shape:", X.shape)


# ==========================================
# CONVERT TO TENSOR
# ==========================================

X_tensor = torch.tensor(X)


# ==========================================
# DATASET
# ==========================================

dataset = TensorDataset(
    X_tensor,
    X_tensor
)


loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# ==========================================
# DEVICE
# ==========================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("Device:", device)


# ==========================================
# MODEL
# ==========================================

model = SensorAutoencoder(
    input_dim=8
).to(device)


# ==========================================
# LOSS
# ==========================================

criterion = nn.MSELoss()


# ==========================================
# OPTIMIZER
# ==========================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ==========================================
# TRAINING
# ==========================================

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0


    for batch_x, _ in loader:

        batch_x = batch_x.to(device)


        # Forward
        embedding, reconstruction = model(
            batch_x
        )


        # Reconstruction loss
        loss = criterion(
            reconstruction,
            batch_x
        )


        # Backpropagation
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        total_loss += (
            loss.item()
            * batch_x.size(0)
        )


    epoch_loss = (
        total_loss / len(dataset)
    )


    if (
        epoch + 1 == 1
        or (epoch + 1) % 10 == 0
    ):

        print(
            f"Epoch "
            f"{epoch + 1}/{EPOCHS} "
            f"- Loss: "
            f"{epoch_loss:.6f}"
        )


# ==========================================
# SAVE
# ==========================================

torch.save(
    model.state_dict(),
    MODEL_PATH
)


print("\nModel saved to:")
print(MODEL_PATH)
