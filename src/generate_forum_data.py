import pandas as pd
import random

students = pd.read_csv("../data/studentInfo.csv")["id_student"]

positive = [
    "I enjoy the course",
    "The lectures are clear",
    "I like the assignments",
    "This module is interesting",
]

neutral = [
    "Working on the assignment",
    "Submitted the quiz",
    "Reading course material",
    "Checking the forum",
]

negative = [
    "I am confused about the lecture",
    "I might drop this course",
    "The assignment is too hard",
    "I am falling behind",
]

rows = []

for student in students:

    for _ in range(random.randint(1, 5)):

        sentiment_type = random.choice(["pos", "neu", "neg"])

        if sentiment_type == "pos":
            text = random.choice(positive)

        elif sentiment_type == "neu":
            text = random.choice(neutral)

        else:
            text = random.choice(negative)

        rows.append([student, text])

df = pd.DataFrame(rows, columns=["id_student", "text"])

df.to_csv("../data/nlp/student_forum_posts.csv", index=False)

print("Forum dataset created")
