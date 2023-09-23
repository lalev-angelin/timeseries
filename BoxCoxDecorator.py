# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 15:35:10 2023

@author: a.lalev
"""

from PredictionMethod import PredictionMethod
from NeuralNetworkPredictionMethod import NeuralNetworkPredictionMethod
from sklearn.linear_model import LinearRegression
from scipy.stats import boxcox
from scipy.special import inv_boxcox

import numpy as np
import json

class  BoxCoxDecorator(PredictionMethod):
    def __init__(self, method):
        self.method = method
        self.data = method.data
        
        
    def toJSON(self):
        str = self.method.toJSON();
        saveData = json.loads(str)
        saveData['boxcox_applied']='true'
        saveData['boxcox_lambda']=self.boxcox_lamda
        return json.dumps(saveData, indent=2)
        
    def predict(self):
         self.originalData = self.data
       
         transformedData = boxcox(self.data)
         
         self.method.data, self.boxcox_lamda = transformedData
         self.prediction = self.method.predict()
         self.prediction = inv_boxcox(self.prediction, self.boxcox_lamda)
         
         return self.prediction.tolist() 
     
    def saveModel(self, filename):
        if isinstance(self.method, NeuralNetworkPredictionMethod):
            self.method.save(filename)