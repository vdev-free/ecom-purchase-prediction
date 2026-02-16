from pathlib import Path
from src.ecom.data_loader import load_raw_data

DATA_PATH = Path('data/raw/online_shoppers.csv')

df = load_raw_data(DATA_PATH)

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isna().sum())

print("\nTarget distribution:")
print(df["Revenue"].value_counts())
