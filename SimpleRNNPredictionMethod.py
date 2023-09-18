#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 10:17:02 2023

@author: ownjo
"""

import numpy as np
from PredictionMethod import PredictionMethod

class SimpleRNNPredictionMethod(PredictionMethod):
    
    def predict(self):
        npTrainData =  np.array(self.trainData)
        scaler = MinMaxScaler(feature_range=(0,1))
        
        rownp = npTrainData.reshape(-1, 1)
        rownp = scaler.fit_transform(rownp)
        rownp = rownp.reshape(-1)

        
        scaledTrainData = 
        
        #### МАЩАБИРАНЕ НА ДАННИТЕ

