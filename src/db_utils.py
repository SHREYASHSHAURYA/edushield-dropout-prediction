import sqlite3
import pandas as pd

DB_PATH = "../data/student_data.db"


def save_to_db(df, table_name="students"):
    conn = sqlite3.connect(DB_PATH)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()


def run_query(query):
    conn = sqlite3.connect(DB_PATH)
    result = pd.read_sql(query, conn)
    conn.close()
    return result
