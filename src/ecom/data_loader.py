from pathlib import Path
import pandas as pd

def load_raw_data(data_path: Path) -> pd.DataFrame:

    df = pd.read_csv(data_path)
    return df