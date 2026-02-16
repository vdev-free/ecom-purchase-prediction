from pathlib import Path
from src.ecom.data_loader import load_raw_data
from src.ecom.data_cleaning import clean_data

RAW_PATH = Path("data/raw/online_shoppers.csv")

df_raw = load_raw_data(RAW_PATH)
df_clean = clean_data(df_raw)

# print("Revenue dtype:", df_clean["Revenue"].dtype)
# print("Revenue unique values:", df_clean["Revenue"].unique())

# print("Weekend dtype:", df_clean["Weekend"].dtype)
# print("Weekend unique values:", df_clean["Weekend"].unique())
# print("Duplicate rows:", df_clean.duplicated().sum())

PROCESSED_PATH = Path("data/processed/online_shoppers_clean.csv")
PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)

df_clean.to_csv(PROCESSED_PATH, index=False)

print("Saved to:", PROCESSED_PATH)

