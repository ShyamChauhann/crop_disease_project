import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# ==========================================
# PATHS
# ==========================================

INPUT_PATH = "/Users/shyamchauhan/Desktop/home/codes/crop_disease_project/dataset/sensor/sensor_data.csv"
OUTPUT_PATH = "/Users/shyamchauhan/Desktop/home/codes/crop_disease_project/dataset/sensor/sensor_processed.csv"


# ==========================================
# CREATE OUTPUT DIRECTORY
# ==========================================

os.makedirs("processed", exist_ok=True)


# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(INPUT_PATH)

print("Original columns:")
print(df.columns.tolist())


# ==========================================
# RENAME COLUMNS
# ==========================================

column_mapping = {
    "Soil Moisture": "soil_moisture",
    "Temperature": "temperature",
    "Air Humidity": "humidity",
    "Pump Data": "pump_data"
}

df = df.rename(columns=column_mapping)


print("\nRenamed columns:")
print(df.columns.tolist())


# ==========================================
# SELECT SENSOR FEATURES
# ==========================================

sensor_columns = [
    "soil_moisture",
    "temperature",
    "humidity",
    "pump_data"
]


# ==========================================
# CONVERT TO NUMERIC
# ==========================================

for column in sensor_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ==========================================
# MISSING VALUE HANDLING
# Linear interpolation
# ==========================================

print("\nMissing values BEFORE interpolation:")

print(
    df[sensor_columns].isnull().sum()
)


df[sensor_columns] = (
    df[sensor_columns]
    .interpolate(method="linear")
)


# Handle values at the beginning/end
df[sensor_columns] = (
    df[sensor_columns]
    .bfill()
    .ffill()
)


print("\nMissing values AFTER interpolation:")

print(
    df[sensor_columns].isnull().sum()
)


# ==========================================
# TEMPORAL FEATURES
# WINDOW = 5
# ==========================================

WINDOW = 5


# Temperature moving average
df["temperature_ma"] = (
    df["temperature"]
    .rolling(
        window=WINDOW,
        min_periods=1
    )
    .mean()
)


# Temperature rolling standard deviation
df["temperature_std"] = (
    df["temperature"]
    .rolling(
        window=WINDOW,
        min_periods=1
    )
    .std()
)


# Humidity moving average
df["humidity_ma"] = (
    df["humidity"]
    .rolling(
        window=WINDOW,
        min_periods=1
    )
    .mean()
)


# Humidity rolling standard deviation
df["humidity_std"] = (
    df["humidity"]
    .rolling(
        window=WINDOW,
        min_periods=1
    )
    .std()
)


# First rows have NaN std because
# there are not enough previous values.

df["temperature_std"] = (
    df["temperature_std"].fillna(0)
)

df["humidity_std"] = (
    df["humidity_std"].fillna(0)
)


# ==========================================
# FINAL FEATURES
# ==========================================

features = [
    "soil_moisture",
    "temperature",
    "humidity",
    "pump_data",
    "temperature_ma",
    "temperature_std",
    "humidity_ma",
    "humidity_std"
]


print("\nFinal features:")
print(features)


# ==========================================
# Z-SCORE NORMALIZATION
# ==========================================

scaler = StandardScaler()

df[features] = scaler.fit_transform(
    df[features]
)


# ==========================================
# SAVE
# ==========================================

df.to_csv(
    OUTPUT_PATH,
    index=False
)


print("\nProcessed dataset saved to:")

print(OUTPUT_PATH)

print("\nFinal shape:")
print(df.shape)

print("\nFirst 5 processed rows:")
print(df.head())