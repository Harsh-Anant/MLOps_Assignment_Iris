from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_predict():
  input = {
    "sepal length (cm)": 7.7,
    "sepal width (cm)": 2.6,
    "petal length (cm)": 6.9,
    "petal width (cm)": 2.3
  }
  response = client.post("/predict", json=input)
  assert response.status_code == 200
  assert "prediction" in response.json()
