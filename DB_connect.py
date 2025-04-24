# DB_connect.py

import pyodbc
import pandas as pd

def connect_to_database():
    conn_str = (
        r"DRIVER={ODBC Driver 17 for SQL Server};"
        r"SERVER=DESKTOP-KEJFO11;"
        r"DATABASE=inventory;"
        r"Trusted_Connection=yes;"
    )
    try:
        return pyodbc.connect(conn_str)
    except pyodbc.Error as e:
        print(f"❌ Connection Error: {e}")
        return None

def run_sql_query(sql_query):
    conn = connect_to_database()
    if conn is None:
        return "❌ Could not establish connection to the database."

    try:
        df = pd.read_sql(sql_query, conn)
        return df
    except Exception as e:
        return f"❌ SQL Error: {e}"
    finally:
        conn.close()
