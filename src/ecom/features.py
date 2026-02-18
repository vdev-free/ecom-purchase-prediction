import pandas as pd

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["TotalPages"] = (
        df["Administrative"] + df["Informational"] + df["ProductRelated"]
    )

    df["TotalDuration"] = (
    df["Administrative_Duration"]
    + df["Informational_Duration"]
    + df["ProductRelated_Duration"]
    )

    df["DurationPerPage"] = df["TotalDuration"] / (df["TotalPages"] + 1)

    return df

