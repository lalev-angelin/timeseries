# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 15:35:10 2023

@author: a.lalev
"""
from PredictionMethod import PredictionMethod
from sklearn.linear_model import LinearRegression
import numpy as np
import json

class DetrendingDecorator(PredictionMethod):
    def __init__(self, method):
        self.method = method
        self.data = method.data
        
        
    def toJSON(self):
        str = self.method.toJSON;
        saveData = json.loads(str)
        saveData['detrend_applied']='true'
        saveData['detrend_type']='linear'
        return json.dumps(saveData, indent=2)
        
    def predict(self):
         self.originalData = self.data
         
         X = np.arange(0, len(self.data))
         X = X.reshape(-1,1)
         
         model = LinearRegression().fit(X, self.data)
         
         self.correction = model.predict(X)
         self.data = self.data - self.correction
         
         self.prediction = self.method.predict()
         
         self.prediction = self.prediction + self.correction
         return self.prediction 