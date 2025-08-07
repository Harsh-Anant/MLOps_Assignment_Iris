# app.py
import mlflow.pyfunc
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import os
import sqlite3
from src.logging_and_monitoring import init_db, log_prediction
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import time
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient



app = FastAPI()

# Prometheus Metrics
REQUEST_COUNT = Counter("prediction_requests_total", "Total number of prediction requests")
REQUEST_LATENCY = Histogram("prediction_request_latency_seconds", "Latency of prediction requests in seconds")

init_db()

# Load the model from MLflow Model Registry

#print(f"MLFLOW_TRACKING_URI from environment: {os.environ.get('MLFLOW_TRACKING_URI')}")
#model = mlflow.pyfunc.load_model("models:/IrisSpeciesClassifier/Production")  # or use 'Production'

# Load the model from MLflow Model Registry

mlflow.set_tracking_uri("file:///app/mlruns")

client = MlflowClient()

versions = client.get_latest_versions(name='IrisSpeciesClassifier')

# Find the one with highest version number (latest)

latest_version = max(versions, key=lambda v: int(v.version))

model = mlflow.pyfunc.load_model(f"models:/IrisSpeciesClassifier/{latest_version.version}")  # or use 'Production'

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
    start_time = time.time()
    REQUEST_COUNT.inc()  # Increment request count
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

    # Record latency
    latency = time.time() - start_time
    REQUEST_LATENCY.observe(latency)
    
    return {"prediction": iris_category}

@app.get("/logs")
def get_logs():
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions ORDER BY timestamp desc")
    rows = cursor.fetchall()
    conn.close()
    return {"logs": rows}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
