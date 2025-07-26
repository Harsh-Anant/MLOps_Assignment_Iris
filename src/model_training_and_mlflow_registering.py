#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import os


# In[2]:


# Load processed data
X_train = pd.read_csv('C:/Users/krish/Downloads/jupyter_projects/data/X_train.csv')
X_test = pd.read_csv('C:/Users/krish/Downloads/jupyter_projects/data/X_test.csv')
y_train = pd.read_csv('C:/Users/krish/Downloads/jupyter_projects/data/y_train.csv').squeeze() # .squeeze() to convert DataFrame to Series
y_test = pd.read_csv('C:/Users/krish/Downloads/jupyter_projects/data/y_test.csv').squeeze()


# In[3]:


mlflow.set_experiment("Iris Classification Experiment")


# In[4]:


mlflow.autolog()


# In[5]:


def train_logistic_regression_model_iris():

    # Set MLflow tracking URI (optional, defaults to ./mlruns)
    # mlflow.set_tracking_uri("http://localhost:5000") # If you run a separate MLflow server

    # --- Train Logistic Regression Model ---
    with mlflow.start_run(run_name="Logistic_Regression_Iris"):
        # Log parameters
        solver = "lbfgs"
        max_iter = 1000
        '''mlflow.log_param("model_name", "Logistic Regression")
        mlflow.log_param("solver", solver)
        mlflow.log_param("max_iter", max_iter)'''

        model_lr = LogisticRegression(solver=solver, max_iter=max_iter)
        model_lr.fit(X_train, y_train)
        y_pred_lr = model_lr.predict(X_test)

        # Log metrics
        accuracy_lr = accuracy_score(y_test, y_pred_lr)
        precision_lr = precision_score(y_test, y_pred_lr, average='weighted')
        recall_lr = recall_score(y_test, y_pred_lr, average='weighted')
        f1_lr = f1_score(y_test, y_pred_lr, average='weighted')

        # Log model
        '''mlflow.sklearn.log_model(model_lr, name="logistic_regression_model", input_example=X_train.head(1))'''

        mlflow.log_metrics({
            "testing_accuracy": accuracy_lr,
            "testing_precision": precision_lr,
            "testing_recall": recall_lr,
            "testing_f1_score": f1_lr
        })
        #print(f"Logistic Regression Accuracy: {accuracy_lr}")


# In[6]:


def train_random_forest_classifier_model_iris():
  # --- Train RandomForestClassifier Model ---
    with mlflow.start_run(run_name="RandomForest_Iris"):
        # Log parameters
        n_estimators = 100
        max_depth = 10
        '''mlflow.log_param("model_name", "Random Forest")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)'''

        model_rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model_rf.fit(X_train, y_train)
        y_pred_rf = model_rf.predict(X_test)

        # Log metrics
        accuracy_rf = accuracy_score(y_test, y_pred_rf)
        precision_rf = precision_score(y_test, y_pred_rf, average='weighted')
        recall_rf = recall_score(y_test, y_pred_rf, average='weighted')
        f1_rf = f1_score(y_test, y_pred_rf, average='weighted')

        #Log model
        '''mlflow.sklearn.log_model(model_rf, name="random_forest_model",input_example=X_train.head(1))'''

        mlflow.log_metrics({
            "testing_accuracy": accuracy_rf,
            "testing_precision": precision_rf,
            "testing_recall": recall_rf,
            "testing_f1_score": f1_rf
        })
        #print(f"Random Forest Accuracy: {accuracy_rf}")

        # Log model


    #print("\nMLflow experiments logged. Run 'mlflow ui' in your terminal to view them.")


# In[8]:


train_logistic_regression_model_iris()


# In[9]:


train_random_forest_classifier_model_iris()


# In[10]:


run_id = "004e3d78b3f5459b96f48affb9760d10"
model_uri=f"runs:/{run_id}/model"
mlflow.register_model(model_uri=model_uri, name="IrisSpeciesClassifier")

