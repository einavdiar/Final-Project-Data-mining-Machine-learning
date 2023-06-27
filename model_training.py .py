# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 21:19:42 2023

@author: einav
"""
import numpy as np
import pandas as pd
import seaborn as sns
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from madlan_model_prep import prepare_data

data = pd.read_excel('output_all_students_Train_v10.xlsx')
Data = prepare_data(data)

X = Data.drop('price', axis=1)
y = Data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = ElasticNet()
model.fit(X_train, y_train)

# Evaluate the performance of the model using cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
mse_scores = -cv_scores
mean_mse = mse_scores.mean()
r2_scores = cross_val_score(model, X_train, y_train, cv=10, scoring='r2')
# Compute average performance metrics on training set
average_mse = mse_scores.mean()
average_r2 = r2_scores.mean()
rmse = np.sqrt(average_mse)
# Fit the model on the entire training set
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate model performance on the test set
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the performance metrics
print("Training Set:")
print('Mean Squared Error:', mean_mse)
print(" R^2:", average_r2)
print("RMSE:", rmse)
print("\nTest Set:")
print("MSE:", mse)
print("R^2:", r2)


import pickle
with open('trained_model.pkl', 'wb') as file:
    pickle.dump(model, file)