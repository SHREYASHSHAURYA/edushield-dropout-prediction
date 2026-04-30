# EduShield: Early Student Dropout Risk Prediction & Intervention System

EduShield is an **explainable machine learning system** designed to identify students at risk of dropping out **early in the semester**.

The system combines **behavioral engagement data, academic performance indicators, and sentiment signals from student forum activity** to predict dropout risk and recommend targeted interventions.

EduShield includes:

• A full **machine learning pipeline**
• **Explainable AI analysis using SHAP**
• An **interactive Streamlit analytics dashboard** for educators and advisors

---

# Problem

Student dropout is a major challenge in higher education and online learning environments.

Traditional approaches often detect dropout **too late**, when intervention is no longer effective.

EduShield aims to:

• Detect dropout risk **early in the learning cycle**
• Explain **why a student is at risk**
• Suggest **targeted interventions**
• Provide educators with **actionable analytics tools**

---

# Key Features

## Early Dropout Prediction

Machine learning models estimate the probability of student dropout using:

• learning engagement metrics
• assessment performance indicators
• behavioral activity patterns
• sentiment signals from student discussions

---

## Explainable AI (SHAP)

Predictions are explained using **SHAP (SHapley Additive Explanations)**.

This allows educators to understand:

• which features contributed to a student's risk score
• how engagement or academic performance affects predictions
• which factors require immediate intervention

---

## Intervention Recommendation Engine

Based on model explanations, the system suggests actionable interventions such as:

• engagement reminders
• tutoring support
• deadline flexibility discussions
• advisor outreach
• emotional or counseling support

---

## Interactive Analytics Dashboard

A Streamlit dashboard provides real-time analytics including:

• cohort-level key metrics (total students, risk distribution, dropout rate)  
• risk segmentation (low / medium / high)  
• feature distribution analysis  
• dropout prediction for individual students  
• explainable AI visualizations (SHAP)  
• global feature importance and impact distributions  
• engagement-based risk simulation  
• cohort risk progression visualization  
• high-risk student monitoring  
• downloadable reports for analysis

---

## Risk Simulation

Educators can simulate behavioral changes (such as increased engagement activity) to observe how a student's **dropout risk may change over time**.

This enables proactive academic support strategies.

---

## Data Engineering Layer

EduShield includes a lightweight data engineering pipeline to support scalable data processing and analysis.

### ETL Pipeline

A structured ETL pipeline was implemented to:

• Load raw student data  
• Clean and preprocess features  
• Transform categorical and behavioral variables  
• Prepare model-ready datasets

This ensures consistency and reproducibility across experiments.

---

### SQL Integration

The processed dataset is stored in a relational database (SQLite), enabling:

• efficient querying of student records  
• cohort-level aggregation and filtering  
• analytical queries for risk segmentation

Example use cases include:

• identifying high-risk student groups  
• computing engagement trends  
• extracting subsets for intervention analysis

---

### Data Validation

Data quality checks are applied before model training:

• missing value detection  
• duplicate record validation  
• range checks for key features (scores, engagement)

This ensures robustness and reliability of model predictions.

---

# Dashboard Overview

The interactive dashboard provides the following analytics and monitoring tools:

• Key cohort metrics

- Total students
- High risk student count
- Average risk score
- Historical dropout rate

• Course dropout distribution analysis
• Highest risk student leaderboard
• Individual student selection and profile view
• Predicted dropout probability for selected student
• Risk gauge visualization
• SHAP explanation of prediction drivers
• Global model feature importance analysis
• SHAP beeswarm feature impact distribution
• Early warning engagement simulation
• Sentiment indicators from forum analysis

- negative sentiment ratio
- negative post count

• Automated intervention recommendations
• Advisor intervention monitoring table for high-risk students
• Course dropout heatmap visualization
• Cohort engagement risk progression simulation

---

## Analytics & Insights

The system generates actionable insights to support decision-making:

• risk segmentation across student populations  
• engagement vs dropout relationship analysis  
• feature distribution summaries  
• cohort-level dropout trends  
• identification of intervention targets

These analytics help educators move from prediction to action.

---

# Machine Learning Models

Multiple models were evaluated for dropout prediction:

• Logistic Regression (Baseline)
• Random Forest
• Logistic Regression with SMOTE (class imbalance handling)
• XGBoost

---

# Model Evaluation

The models were evaluated on a **70 / 30 train-test split**.

Dataset size: **32,593 students**

Test samples: **9,778 students**

| Model                          | Accuracy | Precision (Dropout=1) | Recall (Dropout=1) | F1 Score (Dropout=1) | ROC-AUC   |
| ------------------------------ | -------- | --------------------- | ------------------ | -------------------- | --------- |
| Logistic Regression (Baseline) | 0.86     | 0.77                  | 0.81               | 0.79                 | 0.937     |
| Random Forest                  | 0.88     | 0.78                  | 0.84               | 0.81                 | 0.947     |
| Logistic Regression + SMOTE    | 0.86     | 0.76                  | 0.82               | 0.79                 | 0.933     |
| XGBoost                        | **0.88** | **0.79**              | 0.83               | **0.81**             | **0.950** |

XGBoost achieved the **highest ROC-AUC (0.95)** and strong overall performance.

Therefore **XGBoost was selected as the primary model for the dashboard deployment**.

---

# Dataset

The system uses the **Open University Learning Analytics Dataset (OULAD)**.

The dataset contains:

• student demographics
• course information
• assessment results
• learning activity engagement
• registration activity

Additional engineered features include:

• sentiment indicators derived from simulated forum posts
• behavioral engagement metrics
• temporal activity indicators

---

# Project Structure

```
dropout_predictor
│
├── dashboard
│ └── app.py
│
├── data
│ ├── nlp
│ │ ├── sentiment_features.csv
│ │ └── student_forum_posts.csv
│ │
│ └── final_dataset.csv
│
├── src
│ ├── build_dataset.py
│ ├── explain_model.py
│ ├── generate_forum_data.py
│ ├── intervention_recommender.py
│ ├── load_data.py
│ ├── nlp_sentiment.py
│ ├── prepare_ml_data.py
│ ├── train_baseline_model.py
│ ├── train_random_forest.py
│ ├── train_smote_model.py
│ └── train_xgboost.py
│
├── requirements.txt
└── README.md
```

---

# Running the Project

## Install Dependencies

```
pip install -r requirements.txt
```

---

## Generate NLP Sentiment Features

```
python src/nlp_sentiment.py
```

---

## Build the Machine Learning Dataset

```
python src/build_dataset.py
```

---

## Train the Model

```
python src/train_xgboost.py
```

---

## Launch the Dashboard

```
streamlit run dashboard/app.py
```

---

# Technologies Used

• Python
• Pandas
• Scikit-learn
• XGBoost
• SHAP (Explainable AI)
• Streamlit
• Matplotlib
• Plotly
• SQLite (SQL-based data storage and querying)
• Data Engineering (ETL pipelines, validation)

---

## Export & Reporting

The dashboard supports data export for further analysis:

• full dataset download  
• high-risk student report extraction

This enables integration with external analytics workflows and reporting systems.

---

# Potential Applications

EduShield can be deployed in:

• universities
• online learning platforms
• learning management systems
• academic advising systems

to enable **data-driven student success initiatives and early intervention strategies**.

---

# Future Improvements

Possible extensions include:

• real-time LMS integration
• temporal dropout prediction models
• sequential deep learning models for behavioral patterns
• automated intervention recommendation systems
• institution-level cohort risk monitoring

---

# License

This project is intended for **educational and research purposes**.
