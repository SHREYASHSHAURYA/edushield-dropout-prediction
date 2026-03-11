import streamlit as st
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.set_page_config(page_title="Student Dropout Risk Dashboard", layout="wide")

st.title("Early Student Dropout Risk Predictor")

data = pd.read_csv("data/final_dataset.csv")

data_model = data.drop(
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

data_model = pd.get_dummies(data_model, columns=categorical_cols)

data_model.columns = data_model.columns.str.replace("[", "_", regex=False)
data_model.columns = data_model.columns.str.replace("]", "_", regex=False)
data_model.columns = data_model.columns.str.replace("<", "lt_", regex=False)

X = data_model.drop(columns=["dropout"])
X = X.astype(float)
y = data_model["dropout"]

model = xgb.XGBClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
)

model.fit(X, y)
data["risk_score"] = model.predict_proba(X)[:, 1]

high_risk_count = (data["risk_score"] > 0.7).sum()
medium_risk_count = ((data["risk_score"] >= 0.3) & (data["risk_score"] <= 0.7)).sum()
avg_risk = data["risk_score"].mean()
dropout_rate = data["dropout"].mean()

colA, colB, colC, colD = st.columns(4)

with colA:
    st.metric("Total Students", len(data))

with colB:
    st.metric("High Risk Students", int(high_risk_count))

with colC:
    st.metric("Average Risk Score", f"{avg_risk:.2f}")

with colD:
    st.metric("Historical Dropout Rate", f"{dropout_rate:.2%}")


def risk_category(x):
    if x < 0.3:
        return "Low"
    elif x < 0.6:
        return "Medium"
    else:
        return "High"


data["risk_level"] = data["risk_score"].apply(risk_category)

st.subheader("Course Dropout Distribution")

dropout_counts = data["dropout"].value_counts()

st.bar_chart(dropout_counts.sort_index())

st.subheader("Highest Risk Students")

top_risk = data.sort_values("risk_score", ascending=False).head(50).copy()

top_risk["risk_score"] = top_risk["risk_score"].round(3)

top_risk.index = range(1, len(top_risk) + 1)

st.dataframe(
    top_risk[
        [
            "id_student",
            "code_module",
            "risk_score",
            "risk_level",
            "negative_sentiment_ratio",
            "assessment_submission_count",
        ]
    ]
)

student_ids = data["id_student"].tolist()

st.subheader("Student Selector")

selected_student = st.selectbox("Select Student ID", student_ids)

student_row = data[data["id_student"] == selected_student]

st.subheader("Student Information")

st.dataframe(student_row)

student_model_row = student_row.drop(
    columns=[
        "id_student",
        "code_module",
        "code_presentation",
        "final_result",
        "date_registration",
        "date_unregistration",
    ]
)

student_model_row = pd.get_dummies(student_model_row)

student_model_row = student_model_row.reindex(columns=X.columns, fill_value=0)

student_model_row = student_model_row.astype(float)

risk_prob = model.predict_proba(student_model_row)[0][1]

st.subheader("Predicted Dropout Risk")

st.metric("Dropout Probability", f"{risk_prob:.2f}")

st.subheader("Risk Gauge")

gauge = go.Figure(
    go.Indicator(
        mode="gauge+number",
        value=risk_prob * 100,
        title={"text": "Risk Level"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "red"},
            "steps": [
                {"range": [0, 30], "color": "lightgreen"},
                {"range": [30, 60], "color": "yellow"},
                {"range": [60, 100], "color": "salmon"},
            ],
        },
    )
)

st.plotly_chart(gauge, width="stretch")


@st.cache_resource
def load_explainer():
    return shap.Explainer(model.predict, X)


explainer = load_explainer()

shap_values = explainer(student_model_row)

st.subheader("Risk Explanation (SHAP)")

fig = plt.figure()

shap.plots.waterfall(shap_values[0], max_display=10, show=False)

st.pyplot(fig)

st.subheader("Global Dropout Drivers (Model Insight)")

sample_X = X.sample(500, random_state=42)

sample_shap = explainer(sample_X)

fig2 = plt.figure()

shap.plots.bar(sample_shap, max_display=10, show=False)

st.pyplot(fig2)

st.subheader("Feature Impact Distribution (SHAP Beeswarm)")

sample_X2 = X.sample(500, random_state=42)

beeswarm_values = explainer(sample_X2)

fig4 = plt.figure()

shap.plots.beeswarm(beeswarm_values, max_display=10, show=False)

st.pyplot(fig4)

st.subheader("Early Warning Simulation")

engagement_slider = st.slider(
    "Simulate VLE clicks (first 30 days)",
    0,
    1000,
    int(student_model_row["vle_clicks_30_days"].iloc[0]),
)

simulation_row = student_model_row.copy()

simulation_row["vle_clicks_30_days"] = engagement_slider

simulated_risk = model.predict_proba(simulation_row)[0][1]

st.metric("Simulated Dropout Risk", f"{simulated_risk:.2f}")

st.subheader("Sentiment Indicators")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Negative Sentiment Ratio",
        float(student_row["negative_sentiment_ratio"].iloc[0]),
    )

with col2:
    st.metric("Negative Post Count", int(student_row["negative_post_count"].iloc[0]))


def recommend(feature):

    if feature in [
        "vle_clicks_30_days",
        "active_days",
        "last_activity_day",
        "engagement_decay",
    ]:
        return "Send engagement reminder and advisor outreach"

    if feature in ["avg_score_first_2_assessments", "score_trend"]:
        return "Recommend tutoring support"

    if feature == "late_submission":
        return "Discuss deadline flexibility"

    if feature == "assessment_submission_count":
        return "Encourage assessment participation"

    if feature in ["negative_sentiment_ratio", "negative_post_count"]:
        return "Provide emotional support or counseling"

    return "Advisor review recommended"


shap_importance = abs(shap_values.values[0])

top_index = shap_importance.argsort()[-3:][::-1]

top_features = [X.columns[i] for i in top_index]

st.subheader("Recommended Interventions")

for rec in set([recommend(f) for f in top_features]):
    st.write("-", rec)

st.subheader("Students Requiring Advisor Intervention")

high_risk_students = data[data["risk_score"] > 0.7]

st.dataframe(
    high_risk_students[
        [
            "id_student",
            "code_module",
            "risk_score",
            "negative_post_count",
            "assessment_submission_count",
        ]
    ].head(20)
)

st.subheader("Course Dropout Heatmap")

course_dropout = data.groupby("code_module")["dropout"].mean().reset_index()

course_dropout = course_dropout.pivot_table(
    values="dropout",
    index="code_module",
)

fig, ax = plt.subplots()

ax.imshow(course_dropout, aspect="auto")

ax.set_yticks(range(len(course_dropout.index)))
ax.set_yticklabels(course_dropout.index)

ax.set_title("Dropout Rate by Course")

st.pyplot(fig)

st.subheader("Cohort Risk Progression Simulation")

engagement_levels = list(range(0, 1000, 50))

risk_progression = []

for clicks in engagement_levels:

    sim_row = student_model_row.copy()

    sim_row["vle_clicks_30_days"] = clicks

    risk = model.predict_proba(sim_row)[0][1]

    risk_progression.append(risk)

progress_df = pd.DataFrame(
    {"VLE Clicks": engagement_levels, "Predicted Dropout Risk": risk_progression}
)

fig3, ax = plt.subplots()

ax.plot(progress_df["VLE Clicks"], progress_df["Predicted Dropout Risk"], linewidth=3)

ax.set_xlabel("Engagement (VLE Clicks)")
ax.set_ylabel("Dropout Risk")
ax.set_title("Risk Change with Engagement")

st.pyplot(fig3)
