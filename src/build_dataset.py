import pandas as pd
from db_utils import save_to_db

data_path = "../data/"

student_info = pd.read_csv(data_path + "studentInfo.csv")
student_registration = pd.read_csv(data_path + "studentRegistration.csv")
student_vle = pd.read_csv(data_path + "studentVle.csv")

last_activity = (
    student_vle.groupby(["code_module", "code_presentation", "id_student"])["date"]
    .max()
    .reset_index()
)

last_activity.rename(columns={"date": "last_activity_day"}, inplace=True)

activity_days = (
    student_vle.groupby(["code_module", "code_presentation", "id_student"])["date"]
    .nunique()
    .reset_index()
)

activity_days.rename(columns={"date": "active_days"}, inplace=True)

first_activity = (
    student_vle.groupby(["code_module", "code_presentation", "id_student"])["date"]
    .min()
    .reset_index()
)

first_activity.rename(columns={"date": "first_activity_day"}, inplace=True)

student_assessment = pd.read_csv(data_path + "studentAssessment.csv")
assessments = pd.read_csv(data_path + "assessments.csv")

student_info["dropout"] = student_info["final_result"].apply(
    lambda x: 1 if x == "Withdrawn" else 0
)

dataset = pd.merge(
    student_info,
    student_registration,
    on=["code_module", "code_presentation", "id_student"],
    how="left",
)

vle_total = (
    student_vle.groupby(["code_module", "code_presentation", "id_student"])["sum_click"]
    .sum()
    .reset_index()
)

vle_total.rename(columns={"sum_click": "total_vle_clicks"}, inplace=True)

early_vle = student_vle[student_vle["date"] <= 30]

vle_30 = (
    early_vle.groupby(["code_module", "code_presentation", "id_student"])["sum_click"]
    .sum()
    .reset_index()
)

vle_30.rename(columns={"sum_click": "vle_clicks_30_days"}, inplace=True)

dataset = pd.merge(
    dataset,
    vle_total,
    on=["code_module", "code_presentation", "id_student"],
    how="left",
)

dataset = pd.merge(
    dataset, vle_30, on=["code_module", "code_presentation", "id_student"], how="left"
)

dataset = pd.merge(
    dataset,
    last_activity,
    on=["code_module", "code_presentation", "id_student"],
    how="left",
)

dataset = pd.merge(
    dataset,
    activity_days,
    on=["code_module", "code_presentation", "id_student"],
    how="left",
)

dataset = pd.merge(
    dataset,
    first_activity,
    on=["code_module", "code_presentation", "id_student"],
    how="left",
)

dataset["first_activity_day"] = dataset["first_activity_day"].fillna(0)

dataset["last_activity_day"] = dataset["last_activity_day"].fillna(0)
dataset["active_days"] = dataset["active_days"].fillna(0)

dataset["engagement_decay"] = (
    dataset["last_activity_day"] - dataset["first_activity_day"]
)

dataset["total_vle_clicks"] = dataset["total_vle_clicks"].fillna(0)
dataset["vle_clicks_30_days"] = dataset["vle_clicks_30_days"].fillna(0)

assessment_data = pd.merge(
    student_assessment, assessments, on="id_assessment", how="left"
)

assessment_data["date_submitted"] = pd.to_numeric(
    assessment_data["date_submitted"], errors="coerce"
)

assessment_data["date"] = pd.to_numeric(assessment_data["date"], errors="coerce")
assessment_data["late_submission"] = (
    assessment_data["date_submitted"] > assessment_data["date"]
).astype(int)
late_counts = (
    assessment_data.groupby(["code_module", "code_presentation", "id_student"])[
        "late_submission"
    ]
    .sum()
    .reset_index()
)

dataset = pd.merge(
    dataset,
    late_counts,
    on=["code_module", "code_presentation", "id_student"],
    how="left",
)

dataset["late_submission"] = dataset["late_submission"].fillna(0)
assessment_data["score"] = pd.to_numeric(assessment_data["score"], errors="coerce")

assessment_data = assessment_data.sort_values("date")

first_two = assessment_data.groupby(
    ["code_module", "code_presentation", "id_student"]
).head(2)

score_trend = (
    first_two.groupby(["code_module", "code_presentation", "id_student"])["score"]
    .apply(lambda x: x.iloc[-1] - x.iloc[0] if len(x) > 1 else 0)
    .reset_index()
)

score_trend.rename(columns={"score": "score_trend"}, inplace=True)

avg_scores = (
    first_two.groupby(["code_module", "code_presentation", "id_student"])["score"]
    .mean()
    .reset_index()
)

avg_scores.rename(columns={"score": "avg_score_first_2_assessments"}, inplace=True)
dataset = pd.merge(
    dataset,
    score_trend,
    on=["code_module", "code_presentation", "id_student"],
    how="left",
)

dataset["score_trend"] = dataset["score_trend"].fillna(0)

submission_rate = (
    assessment_data.groupby(["code_module", "code_presentation", "id_student"])["score"]
    .count()
    .reset_index()
)

submission_rate.rename(columns={"score": "assessment_submission_count"}, inplace=True)

dataset = pd.merge(
    dataset,
    avg_scores,
    on=["code_module", "code_presentation", "id_student"],
    how="left",
)

dataset = pd.merge(
    dataset,
    submission_rate,
    on=["code_module", "code_presentation", "id_student"],
    how="left",
)

dataset["avg_score_first_2_assessments"] = dataset[
    "avg_score_first_2_assessments"
].fillna(0)
dataset["assessment_submission_count"] = dataset["assessment_submission_count"].fillna(
    0
)

sentiment = pd.read_csv("../data/nlp/sentiment_features.csv")

dataset = pd.merge(dataset, sentiment, on="id_student", how="left")

dataset["negative_sentiment_ratio"] = dataset["negative_sentiment_ratio"].fillna(0)
dataset["negative_post_count"] = dataset["negative_post_count"].fillna(0)

print("Dataset shape:")
print(dataset.shape)

print("\nNew features:")
print(dataset[["negative_sentiment_ratio", "negative_post_count"]].head())

dataset.to_csv("../data/final_dataset.csv", index=False)

print("\nDataset saved to data/final_dataset.csv")

save_to_db(dataset)

print("Dataset also saved to SQLite DB")
