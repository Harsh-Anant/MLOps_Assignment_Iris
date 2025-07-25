# MLOps_Assignment_Iris

**Terminal Commands**

For checking logged and registered model in mlfow ui
**mlflow ui**

For starting uvicorn server 
**uvicorn app:app --reload**

For building docker image of the prediction service created via FastApi
**docker build -t iris-species-classifier-app .**

For running the container containing the built image
**docker run -p 8000:8000 iris-species-classifier-app**
