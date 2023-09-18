#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 10:17:02 2023

@author: ownjo
"""

import numpy as np
from PredictionMethod import PredictionMethod
#from keras.models import Sequential
#from keras.layers import Dense, SimpleRNN
from sklearn.preprocessing import MinMaxScaler
import sys

class SimpleRNNPredictionMethod(PredictionMethod):
    
    def constructModel(self):
        model = Sequential()
        model.add(SimpleRNN(lenTrainInput, input_shape=(1,1), activation='tanh'))
        model.add(Dense(units=future_predictions, activation='tanh'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        
    
    def predict(self):
        npData = np.array(self.data)
        #print(npData)
       
        scaler = MinMaxScaler(feature_range=(0,1))

        # Мащабиране на обучителното множество
        npScaledData = npData.reshape(-1, 1)
        npScaledData = scaler.fit_transform(npScaledData)
        npScaledData = npScaledData.reshape(-1)
        
        print(npScaledData)
                
        # Не можем да влизаме в тестовото множество
        # a искаме да правим прогноза, обхващаща 
        # numTestPoints периода напред
        npTrainInput = np.array(npScaledData[:-2*self.numTestPoints])
        print(npTrainInput)

        # Оформя изхода на групи от по predictions_plus_gap показатели
        npBatchedOutput = np.array([npScaledData[i:i+self.numTestPoints] for i in range(0, self.numAllPoints-self.numTestPoints+1)])
        print(npBatchedOutput)
        
        npTrainOutput = np.array(npBatchedOutput[1:-self.numTestPoints])
        print(npTrainOutput)
        sys.exit(0)
        
        
        
        #npTrainOutput =  
        # Оформя изхода на групи от по predictions_plus_gap показатели
        #out = np.array([rownp[i:i+future_predictions] for i in range(0, rownp.size-future_predictions+1) ])
        #print("Output dimensions:\n", out.shape)
        #print("Output:\n", out)

    
