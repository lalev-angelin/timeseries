# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 15:35:10 2023

@author: a.lalev
"""
from PredictionMethod import PredictionMethod
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
        