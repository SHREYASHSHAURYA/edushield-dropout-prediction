import pandas as pd
import shap
import xgboost as xgb
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

feature_names = X_test.columns


def recommend(feature):

    if feature in [
        "vle_clicks_30_days",
        "active_days",
        "last_activity_day",
        "engagement_decay",
    ]:
        return "Send engagement reminder and schedule advisor outreach"

    if feature in ["avg_score_first_2_assessments", "score_trend"]:
        return "Recommend academic tutoring support"

    if feature == "late_submission":
        return "Discuss deadline flexibility and time management support"

    if feature == "assessment_submission_count":
        return "Encourage assessment participation and provide academic guidance"

    return "General advisor review"


student_index = 0

contrib = shap_values[student_index][:-1]

importance = pd.Series(contrib, index=feature_names)

top_features = importance.abs().sort_values(ascending=False).head(3)

print("\nStudent risk drivers:\n")

for feature in top_features.index:
    print(feature)

print("\nRecommended interventions:\n")

for feature in top_features.index:
    print(recommend(feature))
