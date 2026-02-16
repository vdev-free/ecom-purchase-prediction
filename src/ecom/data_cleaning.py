import pandas as pd

def clean_data(df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    out = df.copy()
    out['Revenue'] = out['Revenue'].astype(int)
    out['Weekend'] = out['Weekend'].astype(int)

    if drop_duplicates:
        out = out.drop_duplicates()
    return out