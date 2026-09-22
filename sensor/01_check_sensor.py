import pandas as pd

CSV_PATH = "/Users/shyamchauhan/Desktop/home/codes/crop_disease_project/dataset/sensor/sensor_data.csv"


# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv(CSV_PATH)


# ==========================================
# BASIC INFORMATION
# ==========================================

print("\n========== DATASET SHAPE ==========")
print(df.shape)


print("\n========== COLUMNS ==========")
print(df.columns.tolist())


print("\n========== FIRST 5 ROWS ==========")
print(df.head())


print("\n========== DATA TYPES ==========")
print(df.dtypes)


print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())


print("\n========== DUPLICATES ==========")
print(df.duplicated().sum())


print("\n========== STATISTICS ==========")
print(df.describe())