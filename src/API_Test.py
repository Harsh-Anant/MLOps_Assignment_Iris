#!/usr/bin/env python
# coding: utf-8

# In[5]:


import requests

url = "http://127.0.0.1:8000/predict"

input = {
    "sepal length (cm)": 7.7,
    "sepal width (cm)": 2.6,
    "petal length (cm)": 6.9,
    "petal width (cm)": 2.3
}

response = requests.post(url, json=input)
print("Prediction:", (response.json()).get('prediction'))


# In[ ]:




