#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import os

def load_and_preprocess_iris():
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name='species')

    # Basic preprocessing: no missing values, scaling might be needed for some models
    # For simplicity, we'll just split for now
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save processed data
    os.makedirs('data', exist_ok=True)
    X_train.to_csv('data/X_train.csv', index=False)
    X_test.to_csv('data/X_test.csv', index=False)
    y_train.to_csv('data/y_train.csv', index=False)
    y_test.to_csv('data/y_test.csv', index=False)

    print("Iris dataset loaded, preprocessed, and saved to data/")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    load_and_preprocess_iris()


# In[ ]:




