# app.py
import mlflow.pyfunc
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd

app = FastAPI()

# Load the model from MLflow Model Registry

#print(f"MLFLOW_TRACKING_URI from environment: {os.environ.get('MLFLOW_TRACKING_URI')}")
model = mlflow.pyfunc.load_model("models:/IrisSpeciesClassifier/1")  # or use 'Production'

# Define input data schema using Pydantic
#from pydantic import BaseModel, Field

class IrisInput(BaseModel):
    sepal_length_cm: float = Field(..., alias="sepal length (cm)")
    sepal_width_cm: float = Field(..., alias="sepal width (cm)")
    petal_length_cm: float = Field(..., alias="petal length (cm)")
    petal_width_cm: float = Field(..., alias="petal width (cm)")

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
    iris_category = ''
    if prediction == 0:
        iris_category = 'setosa'
    elif prediction == 1:
        iris_category = 'versicolor'
    elif prediction == 2:
        iris_category = 'virginica'
    return {"prediction": iris_category}
