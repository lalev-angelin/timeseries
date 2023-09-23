# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 15:35:10 2023

@author: a.lalev
"""
from PredictionMethod import PredictionMethod
from sklearn.linear_model import LinearRegression
import json

class DetrendingDecorator(PredictionMethod):
    def __init__(self, method, data, numTrainPoints, numSeasons=1):
        super().__init__(self, data, numTrainPoints)
        self.method = method
        
        
    def toJSON(self):
        str = self.method.toJSON;
        saveData = json.loads(str)
        saveData['detrend_applied']='true'
        saveData['detrend_type']='linear'
        return json.dumps(saveData, indent=2)
        
    def predict(self):
         self.originalData = self.data
         model = LinearRegression().fit(range(0, len(self.data)), self.data)
         self.r_sqared = model.r_sqared
         self.correction = model.predict(range(0, len(self.data)))
         self.data = self.data - self.correction
         
         self.prediction = self.method.predict()
         
         self.prediction = self.prediction + self.correction
         return self.prediction 