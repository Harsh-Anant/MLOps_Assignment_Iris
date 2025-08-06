#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sqlite3
from datetime import datetime
import os


# In[4]:


DB_PATH = "db/predictions.db"

def init_db():
    # Create DB only if it doesn't exist
    if not os.path.exists(DB_PATH):
        print("Creating DB")
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sepal_length REAL,
                sepal_width REAL,
                petal_length REAL,
                petal_width REAL,
                prediction INTEGER,
                prediction_label TEXT,
                timestamp TEXT
            )
        ''')
        conn.commit()
        conn.close()

def log_prediction(input_data: dict, prediction: list):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        '''
        INSERT INTO predictions (timestamp, sepal_length, sepal_width, petal_length, petal_width, prediction,prediction_label)
        VALUES (?, ?, ?, ?, ?, ?,?)
        ''',
        (
            str(datetime.now()),
            input_data["sepal_length_cm"],
            input_data["sepal_width_cm"],
            input_data["petal_length_cm"],
            input_data["petal_width_cm"],
            prediction[0],
            prediction[1]
        )
    )
    conn.commit()
    conn.close()


# In[ ]:




