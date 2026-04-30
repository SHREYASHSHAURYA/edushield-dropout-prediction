import streamlit as st
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import matplotlib as mpl
import plotly.graph_objects as go
import base64

st.set_option("client.showErrorDetails", False)
st.set_page_config(page_title="Edushield Dashboard", layout="wide", page_icon="🎓")

# ─── GLOBAL STYLES ────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root palette ── */
:root {
    --bg:         #0F1117;
    --surface:    #161B27;
    --surface2:   #1E2536;
    --border:     #2A3347;
    --accent:     #4F8EF7;
    --accent2:    #7C5CEF;
    --danger:     #F7604F;
    --warn:       #F7B24F;
    --success:    #4FD18B;
    --text:       #E8EDF5;
    --muted:      #8A9BB5;
    --font:       'Sora', sans-serif;
    --mono:       'JetBrains Mono', monospace;
}

/* ── App background ── */
[data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    font-family: var(--font) !important;
}
[data-testid="stHeader"] { background: transparent !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; font-family: var(--font) !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label { color: var(--muted) !important; font-size: 0.78rem !important; letter-spacing: 0.08em; text-transform: uppercase; }

/* ── Global text ── */
html, body, [class*="css"], .stMarkdown, .stText, p, span, div, label {
    font-family: var(--font) !important;
    color: var(--text);
}
h1, h2, h3 { font-family: var(--font) !important; }

/* ── App title ── */
h1 { display: none !important; }

/* ── Subheaders → section labels ── */
h2, h3 {
    color: var(--text) !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
    margin-top: 2rem !important;
    margin-bottom: 1rem !important;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 1.2rem 1.4rem !important;
    transition: border-color 0.2s;
}
[data-testid="metric-container"]:hover { border-color: var(--accent) !important; }
[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    font-weight: 500 !important;
}
[data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-size: 1.7rem !important;
    font-weight: 700 !important;
    font-family: var(--mono) !important;
    letter-spacing: -0.02em !important;
}
[data-testid="stMetricDelta"] { display: none; }

/* ── Dataframe / table ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    overflow: hidden;
}
/* FIX: Make dataframe text visible on dark background */
[data-testid="stDataFrame"] * {
    color: var(--text) !important;
    background-color: transparent !important;
}
.dvn-scroller { background: var(--surface) !important; }
.stDataFrame > div { background: var(--surface) !important; }
/* Dataframe cell and header text */
[data-testid="stDataFrame"] td,
[data-testid="stDataFrame"] th,
[data-testid="stDataFrame"] .cell-text,
[data-testid="stDataFrame"] [class*="cell"],
[data-testid="stDataFrame"] [class*="header"] {
    color: #E8EDF5 !important;
    background: #161B27 !important;
}
/* Target the inner canvas/grid of Streamlit dataframes */
.glide-data-grid-canvas,
.glide-data-grid {
    color: #E8EDF5 !important;
    background: #161B27 !important;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}

/* FIX: Selectbox dropdown popup — force dark bg + light text everywhere */
[data-baseweb="popover"] {
    background: #1E2536 !important;
}
[data-baseweb="popover"] * {
    background: #1E2536 !important;
    color: #E8EDF5 !important;
    font-family: 'Sora', sans-serif !important;
}
[data-baseweb="menu"] {
    background: #1E2536 !important;
    border: 1px solid #2A3347 !important;
    border-radius: 8px !important;
}
[data-baseweb="menu"] li,
[data-baseweb="menu"] [role="option"] {
    background: #1E2536 !important;
    color: #E8EDF5 !important;
    font-family: 'Sora', sans-serif !important;
}
[data-baseweb="menu"] li:hover,
[data-baseweb="menu"] [role="option"]:hover,
[data-baseweb="menu"] [aria-selected="true"] {
    background: #2A3347 !important;
    color: #E8EDF5 !important;
}
/* The actual list items inside the dropdown */
ul[data-baseweb="menu"] li span,
ul[data-baseweb="menu"] li div {
    color: #E8EDF5 !important;
    background: transparent !important;
}
[data-baseweb="select"] > div {
    background: var(--surface2) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}
/* Dropdown list container */
[data-testid="stSelectbox"] ul {
    background: #1E2536 !important;
}
[data-testid="stSelectbox"] ul li {
    color: #E8EDF5 !important;
    background: #1E2536 !important;
}
[data-testid="stSelectbox"] ul li:hover {
    background: #2A3347 !important;
}

/* ── Slider ── */
[data-testid="stSlider"] .stSlider > div { color: var(--accent) !important; }

/* ── Plotly charts ── */
.js-plotly-plot { border-radius: 12px; overflow: hidden; }

/* ── Buttons ── */
button[kind="secondary"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    font-family: var(--font) !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    transition: all 0.2s !important;
}
button[kind="secondary"]:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}

/* ── Pyplot figures ── */
.stImage { border-radius: 10px; overflow: hidden; }

/* ── Markdown links ── */
a { color: var(--accent) !important; text-decoration: none !important; }
a:hover { text-decoration: underline !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }
</style>
""",
    unsafe_allow_html=True,
)


# ─── MATPLOTLIB THEME ─────────────────────────────────────────────────────────
def apply_dark_theme():
    mpl.rcParams.update(
        {
            "figure.facecolor": "#161B27",
            "axes.facecolor": "#161B27",
            "axes.edgecolor": "#2A3347",
            "axes.labelcolor": "#8A9BB5",
            "axes.titlecolor": "#E8EDF5",
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#2A3347",
            "grid.linewidth": 0.5,
            "xtick.color": "#8A9BB5",
            "ytick.color": "#8A9BB5",
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            # FIX: Ensure y-tick (feature name) labels are visible
            "ytick.labelcolor": "#E8EDF5",
            "xtick.labelcolor": "#8A9BB5",
            "text.color": "#E8EDF5",
            "font.family": "sans-serif",
            "figure.dpi": 120,
        }
    )


apply_dark_theme()


# ─── HELPER: small figure ─────────────────────────────────────────────────────
def small_fig(w=6, h=2.8):
    return plt.subplots(figsize=(w, h))


# ─── HEADER BANNER ────────────────────────────────────────────────────────────
st.markdown(
    """
<div style="
    background: linear-gradient(135deg, #161B27 0%, #1E2536 100%);
    border: 1px solid #2A3347;
    border-radius: 16px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1.2rem;
">
    <div style="
        width: 48px; height: 48px;
        background: linear-gradient(135deg, #4F8EF7, #7C5CEF);
        border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        font-size: 22px;
        flex-shrink: 0;
    ">🎓</div>
    <div>
        <div style="font-size:1.35rem; font-weight:700; color:#E8EDF5; letter-spacing:-0.01em;">
            Edushield
        </div>
        <div style="font-size:0.78rem; color:#8A9BB5; margin-top:2px; font-weight:400;">
            XGBoost · SHAP Explainability · Real-time Simulation
        </div>
    </div>
    <div style="margin-left:auto; font-size:0.72rem; color:#8A9BB5; text-align:right; font-family:'JetBrains Mono',monospace;">
        MODEL ACTIVE<br>
        <span style="color:#4FD18B; font-weight:600;">● LIVE</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# ─── DATA + MODEL ─────────────────────────────────────────────────────────────
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
data_model.columns = (
    data_model.columns.str.replace("[", "_", regex=False)
    .str.replace("]", "_", regex=False)
    .str.replace("<", "lt_", regex=False)
)

X = data_model.drop(columns=["dropout"]).astype(float)
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
low_count = (data["risk_score"] < 0.3).sum()
medium_count = medium_risk_count
high_count = high_risk_count


# ─── KPI ROW ──────────────────────────────────────────────────────────────────
colA, colB, colC, colD = st.columns(4)
with colA:
    st.metric("Total Students", len(data))
with colB:
    st.metric("High Risk Students", int(high_risk_count))
with colC:
    st.metric("Average Risk Score", f"{avg_risk:.2f}")
with colD:
    st.metric("Historical Dropout Rate", f"{dropout_rate:.2%}")

# ─── RISK SEGMENTATION ────────────────────────────────────────────────────────
st.subheader("Risk Segmentation Overview")

seg_col1, seg_col2, seg_col3, seg_col4 = st.columns([1, 1, 1, 2])

with seg_col1:
    st.markdown(
        f"""
    <div style="background:#161B27;border:1px solid #2A3347;border-radius:12px;padding:1.2rem;text-align:center;">
        <div style="font-size:0.65rem;letter-spacing:0.12em;text-transform:uppercase;color:#4FD18B;font-weight:600;margin-bottom:6px;">Low Risk</div>
        <div style="font-size:2rem;font-weight:700;color:#4FD18B;font-family:'JetBrains Mono',monospace;">{int(low_count):,}</div>
        <div style="font-size:0.7rem;color:#8A9BB5;margin-top:4px;">score &lt; 0.30</div>
    </div>""",
        unsafe_allow_html=True,
    )

with seg_col2:
    st.markdown(
        f"""
    <div style="background:#161B27;border:1px solid #2A3347;border-radius:12px;padding:1.2rem;text-align:center;">
        <div style="font-size:0.65rem;letter-spacing:0.12em;text-transform:uppercase;color:#F7B24F;font-weight:600;margin-bottom:6px;">Medium Risk</div>
        <div style="font-size:2rem;font-weight:700;color:#F7B24F;font-family:'JetBrains Mono',monospace;">{int(medium_count):,}</div>
        <div style="font-size:0.7rem;color:#8A9BB5;margin-top:4px;">0.30 – 0.70</div>
    </div>""",
        unsafe_allow_html=True,
    )

with seg_col3:
    st.markdown(
        f"""
    <div style="background:#161B27;border:1px solid #2A3347;border-radius:12px;padding:1.2rem;text-align:center;">
        <div style="font-size:0.65rem;letter-spacing:0.12em;text-transform:uppercase;color:#F7604F;font-weight:600;margin-bottom:6px;">High Risk</div>
        <div style="font-size:2rem;font-weight:700;color:#F7604F;font-family:'JetBrains Mono',monospace;">{int(high_count):,}</div>
        <div style="font-size:0.7rem;color:#8A9BB5;margin-top:4px;">score &gt; 0.70</div>
    </div>""",
        unsafe_allow_html=True,
    )

with seg_col4:
    total = low_count + medium_count + high_count
    low_pct = low_count / total * 100
    med_pct = medium_count / total * 100
    high_pct = high_count / total * 100
    st.markdown(
        f"""
    <div style="background:#161B27;border:1px solid #2A3347;border-radius:12px;padding:1.2rem;">
        <div style="font-size:0.65rem;letter-spacing:0.12em;text-transform:uppercase;color:#8A9BB5;font-weight:600;margin-bottom:10px;">Distribution</div>
        <div style="display:flex;height:10px;border-radius:5px;overflow:hidden;margin-bottom:8px;">
            <div style="width:{low_pct:.1f}%;background:#4FD18B;"></div>
            <div style="width:{med_pct:.1f}%;background:#F7B24F;margin:0 2px;"></div>
            <div style="width:{high_pct:.1f}%;background:#F7604F;"></div>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:0.7rem;color:#8A9BB5;font-family:'JetBrains Mono',monospace;">
            <span style="color:#4FD18B;">{low_pct:.1f}%</span>
            <span style="color:#F7B24F;">{med_pct:.1f}%</span>
            <span style="color:#F7604F;">{high_pct:.1f}%</span>
        </div>
    </div>""",
        unsafe_allow_html=True,
    )

st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)

# ─── CHARTS ROW 1 ─────────────────────────────────────────────────────────────
st.subheader("Risk Score Distribution")

ch1, ch2 = st.columns(2)

with ch1:
    fig, ax = small_fig(5.5, 2.8)
    n, bins, patches = ax.hist(data["risk_score"], bins=30, edgecolor="none")
    for patch, left in zip(patches, bins[:-1]):
        if left < 0.3:
            patch.set_facecolor("#4FD18B")
        elif left < 0.7:
            patch.set_facecolor("#F7B24F")
        else:
            patch.set_facecolor("#F7604F")
    ax.set_xlabel("Risk Score")
    ax.set_ylabel("Students")
    ax.set_title("Risk Score Distribution", pad=8)
    plt.tight_layout(pad=0.6)
    st.pyplot(fig)
    plt.close(fig)

with ch2:
    st.subheader("Course Dropout Distribution")
    dropout_counts = data["dropout"].value_counts().sort_index()
    fig2, ax2 = small_fig(5.5, 2.8)
    bars = ax2.bar(
        ["Retained", "Dropped Out"],
        dropout_counts.values,
        color=["#4FD18B", "#F7604F"],
        width=0.5,
        edgecolor="none",
    )
    for bar in bars:
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 20,
            f"{int(bar.get_height()):,}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#8A9BB5",
        )
    ax2.set_title("Dropout vs Retained", pad=8)
    plt.tight_layout(pad=0.6)
    st.pyplot(fig2)
    plt.close(fig2)

# ─── CHARTS ROW 2 ─────────────────────────────────────────────────────────────
st.subheader("Average Feature Values (Top Drivers)")

top_features = [
    "vle_clicks_30_days",
    "avg_score_first_2_assessments",
    "assessment_submission_count",
]
feature_means = data[top_features].mean().round(2)

fig3, ax3 = small_fig(12, 2.4)
bars = ax3.barh(
    [f.replace("_", " ").title() for f in feature_means.index],
    feature_means.values,
    color="#4F8EF7",
    edgecolor="none",
    height=0.5,
)
for bar in bars:
    ax3.text(
        bar.get_width() + 0.5,
        bar.get_y() + bar.get_height() / 2,
        f"{bar.get_width():.1f}",
        va="center",
        fontsize=8,
        color="#8A9BB5",
    )
ax3.set_title("Average Feature Values — Top Predictors", pad=8)
plt.tight_layout(pad=0.6)
st.pyplot(fig3)
plt.close(fig3)

# ─── TOP RISK TABLE ───────────────────────────────────────────────────────────
st.subheader("Highest Risk Students")


def risk_category(x):
    if x < 0.3:
        return "Low"
    elif x < 0.6:
        return "Medium"
    else:
        return "High"


data["risk_level"] = data["risk_score"].apply(risk_category)
top_risk = data.sort_values("risk_score", ascending=False).head(50).copy()
top_risk["risk_score"] = top_risk["risk_score"].round(3)
top_risk.index = range(1, len(top_risk) + 1)


def risk_color_badge(level):
    c = {"High": "#F7604F", "Medium": "#F7B24F", "Low": "#4FD18B"}.get(level, "#8A9BB5")
    return f'<span style="background:{c}22;color:{c};border-radius:20px;padding:2px 10px;font-size:0.7rem;font-weight:600;">{level}</span>'


display_cols = [
    "id_student",
    "code_module",
    "risk_score",
    "risk_level",
    "negative_sentiment_ratio",
    "assessment_submission_count",
]
headers = [
    "Student ID",
    "Module",
    "Risk Score",
    "Risk Level",
    "Neg. Sentiment",
    "Submissions",
]
_tr_header = "".join(
    f'<th style="color:#8A9BB5;font-size:0.65rem;letter-spacing:0.1em;text-transform:uppercase;'
    f'font-weight:600;padding:8px 12px;border-bottom:1px solid #2A3347;text-align:left;">{h}</th>'
    for h in headers
)
_tr_rows = ""
for i, (_, row) in enumerate(top_risk[display_cols].iterrows()):
    bg = "background:#1E253611;" if i % 2 else ""
    _tr_rows += (
        f'<tr style="{bg}border-bottom:1px solid #2A334722;">'
        f'<td style="color:#E8EDF5;padding:7px 12px;font-family:\'JetBrains Mono\',monospace;font-size:0.78rem;">{int(row["id_student"])}</td>'
        f'<td style="color:#8A9BB5;padding:7px 12px;font-size:0.78rem;">{row["code_module"]}</td>'
        f'<td style="color:#4F8EF7;padding:7px 12px;font-family:\'JetBrains Mono\',monospace;font-size:0.78rem;">{row["risk_score"]:.3f}</td>'
        f'<td style="padding:7px 12px;">{risk_color_badge(row["risk_level"])}</td>'
        f'<td style="color:#E8EDF5;padding:7px 12px;font-size:0.78rem;">{row["negative_sentiment_ratio"]:.3f}</td>'
        f'<td style="color:#E8EDF5;padding:7px 12px;font-size:0.78rem;">{int(row["assessment_submission_count"])}</td>'
        f"</tr>"
    )
st.markdown(
    f"""<div style="background:#161B27;border:1px solid #2A3347;border-radius:10px;overflow:hidden;max-height:260px;overflow-y:auto;">
<table style="width:100%;border-collapse:collapse;">
<thead><tr style="background:#1E2536;">{_tr_header}</tr></thead>
<tbody>{_tr_rows}</tbody>
</table></div>""",
    unsafe_allow_html=True,
)

# ─── DIVIDER ──────────────────────────────────────────────────────────────────
st.markdown(
    """
<div style="border-top:1px solid #2A3347;margin:2rem 0 1.5rem;"></div>
<div style="font-size:0.7rem;letter-spacing:0.15em;text-transform:uppercase;color:#8A9BB5;font-weight:600;margin-bottom:1.2rem;">
    ── Individual Student Analysis
</div>
""",
    unsafe_allow_html=True,
)

# ─── STUDENT SELECTOR ─────────────────────────────────────────────────────────
student_ids = data["id_student"].tolist()
st.subheader("Student Selector")
selected_student = st.selectbox(
    "Select Student ID", student_ids, label_visibility="collapsed"
)

student_row = data[data["id_student"] == selected_student]

# ─── STUDENT INFO ─────────────────────────────────────────────────────────────
st.subheader("Student Information")
_si_cols = [c for c in student_row.columns if c not in ["risk_level"]]
_si_header = "".join(
    f'<th style="color:#8A9BB5;font-size:0.62rem;letter-spacing:0.08em;text-transform:uppercase;'
    f'font-weight:600;padding:7px 10px;border-bottom:1px solid #2A3347;text-align:left;white-space:nowrap;">{c}</th>'
    for c in _si_cols
)
_si_vals = "".join(
    f'<td style="color:#E8EDF5;padding:7px 10px;font-size:0.78rem;'
    f"font-family:'JetBrains Mono',monospace;white-space:nowrap;\">{student_row[c].iloc[0]}</td>"
    for c in _si_cols
)
st.markdown(
    f"""<div style="background:#161B27;border:1px solid #2A3347;border-radius:10px;overflow:hidden;overflow-x:auto;margin-bottom:1rem;">
<table style="width:100%;border-collapse:collapse;">
<thead><tr style="background:#1E2536;">{_si_header}</tr></thead>
<tbody><tr>{_si_vals}</tr></tbody>
</table></div>""",
    unsafe_allow_html=True,
)

# ─── MODEL PREP ───────────────────────────────────────────────────────────────
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
student_model_row = student_model_row.reindex(columns=X.columns, fill_value=0).astype(
    float
)

risk_prob = model.predict_proba(student_model_row)[0][1]
risk_color = (
    "#F7604F" if risk_prob > 0.6 else "#F7B24F" if risk_prob > 0.3 else "#4FD18B"
)
risk_label = "HIGH" if risk_prob > 0.6 else "MEDIUM" if risk_prob > 0.3 else "LOW"

# ─── RISK SUMMARY + GAUGE ─────────────────────────────────────────────────────
st.subheader("Predicted Dropout Risk")

risk_col1, risk_col2 = st.columns([1, 2])

with risk_col1:
    st.markdown(
        f"""
    <div style="background:#161B27;border:1px solid {risk_color}33;border-radius:14px;padding:1.6rem;text-align:center;margin-top:4px;">
        <div style="font-size:0.65rem;letter-spacing:0.14em;text-transform:uppercase;color:#8A9BB5;font-weight:600;">Dropout Probability</div>
        <div style="font-size:3rem;font-weight:700;color:{risk_color};font-family:'JetBrains Mono',monospace;margin:8px 0 4px;">{risk_prob:.2f}</div>
        <div style="display:inline-block;background:{risk_color}22;color:{risk_color};border-radius:20px;padding:3px 14px;font-size:0.72rem;font-weight:600;letter-spacing:0.1em;">{risk_label} RISK</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with risk_col2:
    st.subheader("Risk Gauge")
    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk_prob * 100,
            number={
                "suffix": "%",
                "font": {"size": 28, "color": "#E8EDF5", "family": "JetBrains Mono"},
            },
            title={
                "text": "Dropout Risk Level",
                "font": {"size": 12, "color": "#8A9BB5"},
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickcolor": "#8A9BB5",
                    "tickfont": {"size": 9, "color": "#8A9BB5"},
                },
                "bar": {"color": risk_color, "thickness": 0.25},
                "bgcolor": "#1E2536",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 30], "color": "rgba(79, 209, 139, 0.13)"},
                    {"range": [30, 60], "color": "rgba(247, 178, 79, 0.13)"},
                    {"range": [60, 100], "color": "rgba(247, 96, 79, 0.13)"},
                ],
                "threshold": {
                    "line": {"color": risk_color, "width": 3},
                    "thickness": 0.8,
                    "value": risk_prob * 100,
                },
            },
        )
    )
    gauge.update_layout(
        height=240,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="#161B27",
        font_family="Sora, sans-serif",
    )
    st.plotly_chart(gauge, use_container_width=True)


# ─── EXPLAINER ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_explainer():
    return shap.Explainer(model.predict, X)


explainer = load_explainer()
shap_values = explainer(student_model_row)

shap_col1, shap_col2 = st.columns(2)

with shap_col1:
    st.subheader("Risk Explanation (SHAP)")
    fig_s = plt.figure(figsize=(5.5, 3.5))
    fig_s.patch.set_facecolor("#161B27")
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    plt.gcf().patch.set_facecolor("#161B27")
    plt.tight_layout(pad=0.5)
    st.pyplot(fig_s)
    plt.close(fig_s)

with shap_col2:
    st.subheader("Global Dropout Drivers (Model Insight)")
    sample_X = X.sample(500, random_state=42)
    sample_shap = explainer(sample_X)
    fig_g = plt.figure(figsize=(5.5, 3.5))
    fig_g.patch.set_facecolor("#161B27")
    shap.plots.bar(sample_shap, max_display=10, show=False)
    plt.gcf().patch.set_facecolor("#161B27")
    plt.tight_layout(pad=0.5)
    st.pyplot(fig_g)
    plt.close(fig_g)

# ─── BEESWARM ─────────────────────────────────────────────────────────────────
# FIX: Compact figure, explicit left margin so feature labels are never clipped
st.subheader("Feature Impact Distribution (SHAP Beeswarm)")
sample_X2 = X.sample(500, random_state=42)
beeswarm_values = explainer(sample_X2)

fig_b = plt.figure(figsize=(5, 2.6))
fig_b.patch.set_facecolor("#161B27")
shap.plots.beeswarm(beeswarm_values, max_display=8, show=False)
ax_b = plt.gca()
ax_b.set_facecolor("#161B27")
for lbl in ax_b.get_yticklabels():
    lbl.set_color("#E8EDF5")
    lbl.set_fontsize(8)
for lbl in ax_b.get_xticklabels():
    lbl.set_color("#8A9BB5")
    lbl.set_fontsize(7)
plt.gcf().patch.set_facecolor("#161B27")
plt.subplots_adjust(left=0.38, right=0.88, top=0.93, bottom=0.16)
st.pyplot(fig_b, use_container_width=False)
plt.close(fig_b)

# ─── SIMULATION ───────────────────────────────────────────────────────────────
st.markdown(
    """
<div style="border-top:1px solid #2A3347;margin:2rem 0 1.5rem;"></div>
<div style="font-size:0.7rem;letter-spacing:0.15em;text-transform:uppercase;color:#8A9BB5;font-weight:600;margin-bottom:1.2rem;">
    ── Simulation & Intervention Tools
</div>
""",
    unsafe_allow_html=True,
)

st.subheader("Early Warning Simulation")

sim_col1, sim_col2 = st.columns([2, 1])

with sim_col1:
    engagement_slider = st.slider(
        "Simulate VLE clicks (first 30 days)",
        0,
        1000,
        int(student_model_row["vle_clicks_30_days"].iloc[0]),
    )

simulation_row = student_model_row.copy()
simulation_row["vle_clicks_30_days"] = engagement_slider
simulated_risk = model.predict_proba(simulation_row)[0][1]
sim_color = (
    "#F7604F"
    if simulated_risk > 0.6
    else "#F7B24F" if simulated_risk > 0.3 else "#4FD18B"
)

with sim_col2:
    st.markdown(
        f"""
    <div style="background:#161B27;border:1px solid {sim_color}44;border-radius:12px;padding:1rem 1.2rem;margin-top:8px;">
        <div style="font-size:0.65rem;letter-spacing:0.12em;text-transform:uppercase;color:#8A9BB5;font-weight:600;">Simulated Risk</div>
        <div style="font-size:2.2rem;font-weight:700;color:{sim_color};font-family:'JetBrains Mono',monospace;margin-top:4px;">{simulated_risk:.2f}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

# ─── SENTIMENT ────────────────────────────────────────────────────────────────
st.subheader("Sentiment Indicators")

sent_col1, sent_col2 = st.columns(2)
neg_ratio = float(student_row["negative_sentiment_ratio"].iloc[0])
neg_posts = int(student_row["negative_post_count"].iloc[0])

with sent_col1:
    bar_w = int(neg_ratio * 100)
    st.markdown(
        f"""
    <div style="background:#161B27;border:1px solid #2A3347;border-radius:12px;padding:1.2rem;">
        <div style="font-size:0.65rem;letter-spacing:0.12em;text-transform:uppercase;color:#8A9BB5;font-weight:600;margin-bottom:8px;">Negative Sentiment Ratio</div>
        <div style="font-size:1.8rem;font-weight:700;color:#F7604F;font-family:'JetBrains Mono',monospace;">{neg_ratio:.3f}</div>
        <div style="height:4px;background:#2A3347;border-radius:2px;margin-top:10px;">
            <div style="height:100%;width:{bar_w}%;background:#F7604F;border-radius:2px;"></div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with sent_col2:
    st.markdown(
        f"""
    <div style="background:#161B27;border:1px solid #2A3347;border-radius:12px;padding:1.2rem;">
        <div style="font-size:0.65rem;letter-spacing:0.12em;text-transform:uppercase;color:#8A9BB5;font-weight:600;margin-bottom:8px;">Negative Post Count</div>
        <div style="font-size:1.8rem;font-weight:700;color:#F7B24F;font-family:'JetBrains Mono',monospace;">{neg_posts}</div>
        <div style="height:4px;background:#2A3347;border-radius:2px;margin-top:10px;">
            <div style="height:100%;width:{min(neg_posts, 100)}%;background:#F7B24F;border-radius:2px;"></div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


# ─── INTERVENTIONS ────────────────────────────────────────────────────────────
def recommend(feature):
    if feature in [
        "vle_clicks_30_days",
        "active_days",
        "last_activity_day",
        "engagement_decay",
    ]:
        return (
            "📣",
            "Engagement Reminder",
            "Send engagement reminder and advisor outreach",
        )
    if feature in ["avg_score_first_2_assessments", "score_trend"]:
        return ("📚", "Academic Support", "Recommend tutoring support")
    if feature == "late_submission":
        return ("📅", "Deadline Support", "Discuss deadline flexibility")
    if feature == "assessment_submission_count":
        return ("✅", "Assessment Push", "Encourage assessment participation")
    if feature in ["negative_sentiment_ratio", "negative_post_count"]:
        return ("💬", "Wellbeing Check", "Provide emotional support or counseling")
    return ("👤", "Advisor Review", "Advisor review recommended")


shap_importance = abs(shap_values.values[0])
top_index = shap_importance.argsort()[-3:][::-1]
top_feats = [X.columns[i] for i in top_index]
recs = list({recommend(f) for f in top_feats})

st.subheader("Recommended Interventions")

rec_cols = st.columns(len(recs))
for col, (icon, title, desc) in zip(rec_cols, recs):
    with col:
        st.markdown(
            f"""
        <div style="background:#161B27;border:1px solid #2A3347;border-radius:12px;padding:1.1rem;height:100%;">
            <div style="font-size:1.3rem;margin-bottom:6px;">{icon}</div>
            <div style="font-size:0.8rem;font-weight:600;color:#E8EDF5;margin-bottom:4px;">{title}</div>
            <div style="font-size:0.72rem;color:#8A9BB5;line-height:1.4;">{desc}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

# ─── ADVISOR TABLE ────────────────────────────────────────────────────────────
st.subheader("Students Requiring Advisor Intervention")

high_risk_students = data[data["risk_score"] > 0.7]

# FIX: HTML table for full visibility on dark background
_adv_cols = [
    "id_student",
    "code_module",
    "risk_score",
    "negative_post_count",
    "assessment_submission_count",
]
_adv_headers = ["Student ID", "Module", "Risk Score", "Neg. Posts", "Submissions"]
_adv_header_html = "".join(
    f'<th style="color:#8A9BB5;font-size:0.65rem;letter-spacing:0.1em;text-transform:uppercase;'
    f'font-weight:600;padding:8px 12px;border-bottom:1px solid #2A3347;text-align:left;">{h}</th>'
    for h in _adv_headers
)
_adv_rows_html = ""
for i, (_, row) in enumerate(high_risk_students[_adv_cols].head(20).iterrows()):
    bg = "background:#1E253611;" if i % 2 else ""
    _adv_rows_html += (
        f'<tr style="{bg}border-bottom:1px solid #2A334722;">'
        f'<td style="color:#E8EDF5;padding:7px 12px;font-family:\'JetBrains Mono\',monospace;font-size:0.78rem;">{int(row["id_student"])}</td>'
        f'<td style="color:#8A9BB5;padding:7px 12px;font-size:0.78rem;">{row["code_module"]}</td>'
        f'<td style="color:#F7604F;padding:7px 12px;font-family:\'JetBrains Mono\',monospace;font-size:0.78rem;">{row["risk_score"]:.3f}</td>'
        f'<td style="color:#E8EDF5;padding:7px 12px;font-size:0.78rem;">{int(row["negative_post_count"])}</td>'
        f'<td style="color:#E8EDF5;padding:7px 12px;font-size:0.78rem;">{int(row["assessment_submission_count"])}</td>'
        f"</tr>"
    )
st.markdown(
    f"""<div style="background:#161B27;border:1px solid #2A3347;border-radius:10px;overflow:hidden;max-height:240px;overflow-y:auto;">
<table style="width:100%;border-collapse:collapse;">
<thead><tr style="background:#1E2536;">{_adv_header_html}</tr></thead>
<tbody>{_adv_rows_html}</tbody>
</table></div>""",
    unsafe_allow_html=True,
)

# ─── DOWNLOADS ────────────────────────────────────────────────────────────────
st.subheader("Download Reports")


def create_download_link(df, filename, label, color="#4F8EF7"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = (
        f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="'
        f"display:inline-block;background:#161B27;border:1px solid {color}44;"
        f"color:{color};border-radius:8px;padding:0.55rem 1.1rem;font-size:0.8rem;"
        f'font-weight:500;text-decoration:none;transition:all 0.2s;">'
        f"⬇ {label}</a>"
    )
    return href


dl1, dl2 = st.columns(2)
with dl1:
    st.markdown(
        create_download_link(data, "student_dropout_full.csv", "Download Full Dataset"),
        unsafe_allow_html=True,
    )
with dl2:
    st.markdown(
        create_download_link(
            high_risk_students,
            "high_risk_students.csv",
            "Download High Risk Students",
            "#F7604F",
        ),
        unsafe_allow_html=True,
    )

# ─── HEATMAP ──────────────────────────────────────────────────────────────────
st.subheader("Course Dropout Heatmap")

course_dropout = data.groupby("code_module")["dropout"].mean().reset_index()
course_dropout_pivot = course_dropout.pivot_table(values="dropout", index="code_module")

fig_h, ax_h = plt.subplots(figsize=(12, max(2.5, len(course_dropout_pivot) * 0.45)))
im = ax_h.imshow(course_dropout_pivot, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1)
ax_h.set_yticks(range(len(course_dropout_pivot.index)))
ax_h.set_yticklabels(course_dropout_pivot.index, fontsize=9)
ax_h.set_xticks([])
ax_h.set_title("Dropout Rate by Course", pad=8)
plt.colorbar(im, ax=ax_h, label="Dropout Rate", fraction=0.015, pad=0.02)
plt.tight_layout(pad=0.6)
st.pyplot(fig_h)
plt.close(fig_h)

# ─── COHORT SIMULATION ────────────────────────────────────────────────────────
st.subheader("Cohort Risk Progression Simulation")

engagement_levels = list(range(0, 1000, 50))
risk_progression = []
for clicks in engagement_levels:
    sim_row = student_model_row.copy()
    sim_row["vle_clicks_30_days"] = clicks
    risk_progression.append(model.predict_proba(sim_row)[0][1])

progress_df = pd.DataFrame(
    {"VLE Clicks": engagement_levels, "Predicted Dropout Risk": risk_progression}
)

fig_p, ax_p = small_fig(12, 2.8)
ax_p.fill_between(
    progress_df["VLE Clicks"],
    progress_df["Predicted Dropout Risk"],
    alpha=0.15,
    color="#4F8EF7",
)
ax_p.plot(
    progress_df["VLE Clicks"],
    progress_df["Predicted Dropout Risk"],
    linewidth=2.5,
    color="#4F8EF7",
)
ax_p.axhline(
    0.7,
    color="#F7604F",
    linewidth=1,
    linestyle="--",
    alpha=0.6,
    label="High risk threshold",
)
ax_p.axhline(
    0.3,
    color="#4FD18B",
    linewidth=1,
    linestyle="--",
    alpha=0.6,
    label="Low risk threshold",
)
ax_p.set_xlabel("Engagement (VLE Clicks)")
ax_p.set_ylabel("Dropout Risk")
ax_p.set_title("Risk Change with Engagement", pad=8)
ax_p.legend(fontsize=8, framealpha=0, labelcolor="#8A9BB5")
plt.tight_layout(pad=0.6)
st.pyplot(fig_p)
plt.close(fig_p)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown(
    """
<div style="border-top:1px solid #2A3347;margin-top:3rem;padding:1.2rem 0 0.5rem;text-align:center;">
    <span style="font-size:0.7rem;color:#8A9BB5;letter-spacing:0.08em;">
        Student Dropout Risk Predictor &nbsp;·&nbsp; XGBoost + SHAP &nbsp;·&nbsp; Built for early academic intervention
    </span>
</div>
""",
    unsafe_allow_html=True,
)
