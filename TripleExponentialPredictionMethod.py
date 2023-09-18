#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 09:39:35 2023

@author: ownjo
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 09:26:35 2023

@author: ownjo
"""

from PredictionMethod import PredictionMethod
from statsmodels.tsa.api import ExponentialSmoothing
import numpy as np

class TripleExponentialPredictionMethod(PredictionMethod):
    
    def __init__(self, data, numTrainPoints, numSeasons=1, params=(None, None, None)):
        super().__init__(data, numTrainPoints, numSeasons)
        self.alpha = params[0]
        self.beta = params[1]
        self.gamma = params[2]
        
        # Или двата параметъра, или нищо. Който иска просто изглаждане, има си клас.
        if self.alpha==None:
            assert self.beta==None
            assert self.gamma==None
        if self.alpha!=None:
            assert self.beta!=None
            assert self.gamma!=None
            
            
    def predict(self):
        
        npdata = np.array(self.trainData)
        
        if (self.alpha==None):
            self.hwresults = ExponentialSmoothing(npdata, initialization_method='estimated',  trend="add", 
            seasonal="add", seasonal_periods=self.numSeasons).fit(optimized=True)    
        else:
            self.hwresults = ExponentialSmoothing(npdata, initialization_method='estimated',  trend="add", 
            seasonal="add", seasonal_periods=self.numSeasons).fit(smoothing_level=self.alpha, trend_level=self.beta, smoothing_seasonal=self.gamma, optimized=False)
            

        self.prediction = self.hwresults.predict(start=0, end=self.numAllPoints)
        return self.prediction