#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# In[4]:


url = 'C://Users//Asna maheen//winequality-red.csv'
df = pd.read_csv(url, delimiter=';')
print(df)


# In[5]:


# Split the data into features (X) and target variable (y)
X = df.drop('quality', axis=1)
y = df['quality']


# In[6]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[7]:


# Create a linear regression model
model = LinearRegression()


# In[8]:


# Train the model
model.fit(X_train, y_train)


# In[9]:


# Make predictions on the test set
y_pred = model.predict(X_test)


# In[10]:


# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# In[11]:


# Print the evaluation results
print("Mean Squared Error:", mse)
print("Coefficient of Determination (R^2):", r2)


# In[ ]:




