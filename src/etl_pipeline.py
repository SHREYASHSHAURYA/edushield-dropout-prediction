import pandas as pd
from data_validation import validate_data


def load_data(path):
    return pd.read_csv(path)


def clean_data(df):
    df = df.fillna(0)
    return df


def transform_data(df):
    df["engagement_ratio"] = df["vle_clicks_30_days"] / (df["active_days"] + 1)
    return df


def run_etl(path):
    df = load_data(path)
    df = validate_data(df)
    df = clean_data(df)
    df = transform_data(df)
    return df
