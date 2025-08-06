# app.py
import mlflow.pyfunc
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import os
import sqlite3
from src.logging_and_monitoring import init_db, log_prediction


app = FastAPI()

init_db()

# Load the model from MLflow Model Registry

#print(f"MLFLOW_TRACKING_URI from environment: {os.environ.get('MLFLOW_TRACKING_URI')}")
model = mlflow.pyfunc.load_model("models:/IrisSpeciesClassifier/1")  # or use 'Production'

# Define input data schema using Pydantic

class IrisInput(BaseModel):
    sepal_length_cm: float = Field(..., alias="sepal length (cm)",ge=0, le=10, description="Range: 0 to 10 cm")
    sepal_width_cm: float = Field(..., alias="sepal width (cm)", ge=0, le=10, description="Range: 0 to 10 cm")
    petal_length_cm: float = Field(..., alias="petal length (cm)", ge=0, le=10, description="Range: 0 to 10 cm")
    petal_width_cm: float = Field(..., alias="petal width (cm)", ge=0, le=10, description="Range: 0 to 10 cm")

class Config:
    allow_population_by_field_name = True  # Needed when using aliases
    
@app.get("/")
def home():
    return {"message": "FastAPI is running!"}
    
@app.post("/predict")
def predict_species(data: IrisInput):
    print(data)
    input_dict = {
        "sepal length (cm)": data.sepal_length_cm,
        "sepal width (cm)": data.sepal_width_cm,
        "petal length (cm)": data.petal_length_cm,
        "petal width (cm)": data.petal_width_cm
    }
    input_df = pd.DataFrame([input_dict])
    prediction = model.predict(input_df)
    
    prediction = prediction.tolist()[0]
    
    iris_category = 'Invalid Input!, make sure each of the four attributes are there and their values ranges from 0 to 10 in cm.'
    
    if prediction == 0:
        iris_category = 'setosa'
    elif prediction == 1:
        iris_category = 'versicolor'
    elif prediction == 2:
        iris_category = 'virginica'
        
    input_to_log = {
        "sepal_length_cm": data.sepal_length_cm,
        "sepal_width_cm": data.sepal_width_cm,
        "petal_length_cm": data.petal_length_cm,
        "petal_width_cm": data.petal_width_cm
    }  
    log_prediction(input_to_log, [prediction,iris_category])
    return {"prediction": iris_category}

@app.get("/logs")
def get_logs():
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions ORDER BY timestamp desc")
    rows = cursor.fetchall()
    conn.close()
    return {"logs": rows}
