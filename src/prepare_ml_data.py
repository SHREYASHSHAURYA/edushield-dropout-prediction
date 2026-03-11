import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("../data/final_dataset.csv")

data = data.drop(
    columns=[
        "id_student",
        "code_module",
        "code_presentation",
        "final_result",
        "date_registration",
        "date_unregistration",
    ]
)

categorical_cols = [
    "gender",
    "region",
    "highest_education",
    "imd_band",
    "age_band",
    "disability",
]

data = pd.get_dummies(data, columns=categorical_cols)

X = data.drop(columns=["dropout"])
y = data["dropout"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

print("\nDropout ratio train:")
print(y_train.value_counts(normalize=True))

print("\nDropout ratio test:")
print(y_test.value_counts(normalize=True))
