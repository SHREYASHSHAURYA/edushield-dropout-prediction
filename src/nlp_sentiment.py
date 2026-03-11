import pandas as pd
from transformers import pipeline

df = pd.read_csv("../data/nlp/student_forum_posts.csv")

sentiment_model = pipeline("sentiment-analysis")

predictions = sentiment_model(df["text"].tolist(), batch_size=64)
df["sentiment"] = [p["label"] for p in predictions]

df["negative"] = (df["sentiment"] == "NEGATIVE").astype(int)

sentiment_features = (
    df.groupby("id_student")["negative"].agg(["mean", "sum"]).reset_index()
)

sentiment_features.columns = [
    "id_student",
    "negative_sentiment_ratio",
    "negative_post_count",
]

sentiment_features.to_csv("../data/nlp/sentiment_features.csv", index=False)

print(sentiment_features.head())
