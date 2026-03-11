import pandas as pd
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
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

data.columns = data.columns.str.replace("[", "_", regex=False)
data.columns = data.columns.str.replace("]", "_", regex=False)
data.columns = data.columns.str.replace("<", "lt_", regex=False)

X = data.drop(columns=["dropout"])
y = data["dropout"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

model = xgb.XGBClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
)

model.fit(X_train, y_train)

dmatrix = xgb.DMatrix(X_test)

shap_values = model.get_booster().predict(dmatrix, pred_contribs=True)

shap.summary_plot(shap_values[:, :-1], X_test)
