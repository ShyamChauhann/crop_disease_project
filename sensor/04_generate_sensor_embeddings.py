import numpy as np
import pandas as pd

import torch

# from sensor.sensor_model import SensorAutoencoder
from sensor_model import SensorAutoencoder


# ==========================================
# PATHS
# ==========================================

DATA_PATH = "/Users/shyamchauhan/Desktop/home/codes/crop_disease_project/dataset/sensor/sensor_processed.csv"

MODEL_PATH = "/Users/shyamchauhan/Desktop/home/codes/crop_disease_project/models/sensor_autoencoder.pth"

OUTPUT_PATH = "/Users/shyamchauhan/Desktop/home/codes/crop_disease_project/embeddings/sensor_embeddings.npy"


# ==========================================
# FEATURES
# ==========================================

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


# ==========================================
# DEVICE
# ==========================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(DATA_PATH)


X = df[FEATURES].values.astype(
    np.float32
)


X_tensor = torch.tensor(X).to(device)


# ==========================================
# LOAD MODEL
# ==========================================

model = SensorAutoencoder(
    input_dim=8
).to(device)


model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)


model.eval()


# ==========================================
# GENERATE EMBEDDINGS
# ==========================================

with torch.no_grad():

    embeddings, _ = model(
        X_tensor
    )


embeddings = (
    embeddings
    .cpu()
    .numpy()
)


# ==========================================
# CREATE DIRECTORY
# ==========================================

import os

os.makedirs(
    "embeddings",
    exist_ok=True
)


# ==========================================
# SAVE
# ==========================================

np.save(
    OUTPUT_PATH,
    embeddings
)


print("Embedding shape:")
print(embeddings.shape)

print("\nSaved:")
print(OUTPUT_PATH)