# EduShield: Early Student Dropout Risk Prediction & Intervention System

EduShield is an explainable machine learning system designed to identify students at risk of dropping out **early in the semester**.
It combines behavioral engagement data, academic performance indicators, and forum sentiment signals to predict dropout risk and recommend targeted interventions.

The system includes a full **machine learning pipeline**, **explainable AI analysis**, and an **interactive Streamlit dashboard** for educators and advisors.

---

## Problem

Student dropout is a major challenge in higher education and online learning platforms.
Traditional approaches often detect dropout **too late**, when intervention is no longer effective.

EduShield aims to:

- Detect dropout risk **early**
- Explain **why** a student is at risk
- Suggest **practical interventions**
- Provide educators with an **interactive analytics dashboard**

---

## Key Features

### Early Dropout Prediction

Uses machine learning models to predict the probability of a student dropping out based on:

- engagement metrics
- assessment performance
- learning activity patterns
- sentiment indicators

---

### Explainable AI (SHAP)

Model predictions are explained using **SHAP values**, allowing educators to understand:

- which features contributed to the risk
- how student behavior affects predictions
- which factors require intervention

---

### Intervention Recommendation Engine

Based on model explanations, the system suggests actions such as:

- engagement reminders
- tutoring support
- deadline flexibility
- advisor review
- emotional or counseling support

---

### Interactive Analytics Dashboard

A Streamlit dashboard provides real-time insights including:

- dropout risk scores
- explainable model predictions
- cohort risk visualization
- engagement simulations
- high-risk student identification

---

### Risk Simulation

Educators can simulate behavioral changes such as increased learning activity to see how a student's risk level may change.

---

## Dashboard Overview

The dashboard includes:

- Course dropout distribution
- Student risk prediction
- SHAP explanation plots
- Feature importance analysis
- Risk simulation tools
- Intervention recommendations
- High-risk student monitoring

---

## Project Structure

```
dropout_predictor
│
├── dashboard
│   └── app.py                # Streamlit dashboard
│
├── data
│   ├── nlp
│   │   ├── sentiment_features.csv
│   │   └── student_forum_posts.csv
│   │
│   └── final_dataset.csv
│
├── src
│   ├── build_dataset.py
│   ├── explain_model.py
│   ├── generate_forum_data.py
│   ├── intervention_recommender.py
│   ├── load_data.py
│   ├── nlp_sentiment.py
│   ├── prepare_ml_data.py
│   ├── train_baseline_model.py
│   ├── train_random_forest.py
│   ├── train_smote_model.py
│   └── train_xgboost.py
│
├── requirements.txt
└── README.md
```

---

## Machine Learning Models

The project explores multiple models for dropout prediction:

- Logistic Regression (baseline)
- Random Forest
- XGBoost
- SMOTE-enhanced models for class imbalance

The final dashboard uses **XGBoost** due to strong performance and interpretability with SHAP.

---

## Dataset

The project uses the **Open University Learning Analytics Dataset (OULAD)**, which contains:

- student demographics
- course information
- assessment results
- engagement with learning resources
- registration activity

Additional features include:

- sentiment indicators from simulated forum posts
- behavioral engagement metrics
- derived learning activity features

---

## Running the Project

### 1 Install dependencies

```
pip install -r requirements.txt
```

---

### 2 Generate NLP sentiment features

```
python src/nlp_sentiment.py
```

---

### 3 Build the machine learning dataset

```
python src/build_dataset.py
```

---

### 4 Train the model

```
python src/train_xgboost.py
```

---

### 5 Launch the dashboard

```
streamlit run dashboard/app.py
```

---

## Technologies Used

- Python
- Pandas
- Scikit-learn
- XGBoost
- SHAP (Explainable AI)
- Streamlit
- Matplotlib
- Plotly

---

## Potential Applications

This system can be applied to:

- universities
- online learning platforms
- learning management systems
- academic advising systems

to enable **data-driven student support and early intervention strategies**.

---

## Future Improvements

Possible extensions include:

- real-time LMS integration
- temporal dropout prediction models
- deep learning for behavioral patterns
- automated intervention planning
- cohort-level risk monitoring

---

## License

This project is intended for educational and research purposes.
