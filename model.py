# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 14:47:43 2021

@author: Susan Varghese
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LinearRegression


data = pd.read_csv(r'E:\19AI718 IoT for AI\ML Model Deployment\SystPress.csv')
X = data.iloc[:, 0:2]
Y = data.iloc[:, 2:]
regressor=LinearRegression()
regressor.fit(X,Y)

# saving model to disk
pickle.dump(regressor,open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
print(model.predict([[40,187]]))
