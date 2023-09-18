#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 06:24:48 2023

@author: ownjo
"""

from PredictionMethod import PredictionMethod
from statsmodels.tsa.api import SimpleExpSmoothing
import numpy as np

class ExponentialPredictionMethod(PredictionMethod):
    
    def __init__(self, data, numTrainPoints, numSeasons=1, alpha=None):
        super().__init__(data, numTrainPoints, numSeasons)
        self.alpha = alpha
    
    def predict(self):
        
        npdata = np.array(self.trainData)
        
        if (self.alpha==None):
            print("NoAlpha")
            self.hwresults = SimpleExpSmoothing(npdata, initialization_method='estimated').fit()
        else:
            print("Alpha")
            self.hwresults = SimpleExpSmoothing(npdata, initialization_method='heuristic').fit(smoothing_level=self.alpha)

        self.prediction = self.hwresults.predict(start=0, end=self.numAllPoints)
        return self.prediction