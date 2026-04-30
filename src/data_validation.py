import pandas as pd


def validate_data(df):
    print("\nRunning Data Validation...")

    missing = df.isnull().sum().sum()
    print(f"Missing values: {missing}")

    duplicates = df.duplicated().sum()
    print(f"Duplicate rows: {duplicates}")

    if "vle_clicks_30_days" in df.columns:
        invalid_clicks = (df["vle_clicks_30_days"] < 0).sum()
        print(f"Invalid VLE clicks (<0): {invalid_clicks}")

    if "avg_score_first_2_assessments" in df.columns:
        invalid_scores = (
            (df["avg_score_first_2_assessments"] < 0)
            | (df["avg_score_first_2_assessments"] > 100)
        ).sum()
        print(f"Invalid scores (<0 or >100): {invalid_scores}")

    print("Data validation completed.\n")

    return df
