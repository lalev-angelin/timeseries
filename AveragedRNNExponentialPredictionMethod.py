#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 19:01:45 2023

@author: ownjo
"""

import numpy as np
from PredictionMethod import PredictionMethod
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from sklearn.preprocessing import MinMaxScaler
import sys
import cmath
from DoubleExponentialPredictionMethod import DoubleExponentialPredictionMethod 
from TripleExponentialPredictionMethod import TripleExponentialPredictionMethod


class AveragedRNNExponentialPredictionMethod(PredictionMethod):
    
    def constructModel(self, rnn1NeuronCount):
        model = Sequential()
        model.add(SimpleRNN(rnn1NeuronCount, input_shape=(1,1), activation='tanh'))
        model.add(Dense(units=self.numTestPoints, activation='tanh'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model
    
    def predict(self):
               
        if self.numSeasons==1: 
            smooth = DoubleExponentialPredictionMethod(self.data, self.numTrainPoints)        
        else:
            smooth = TripleExponentialPredictionMethod(self.data, self.numTrainPoints, numSeasons=self.numSeasons)
        
        smoothData = smooth.predict()
        
        npData = np.array(self.data)
        #print(npData)
       
        scaler = MinMaxScaler(feature_range=(-1,1))

        # Мащабиране на обучителното множество
        npScaledData = npData.reshape(-1, 1)
        npScaledData = scaler.fit_transform(npScaledData)
        npScaledData = npScaledData.reshape(-1)
        
        # Не можем да влизаме в тестовото множество,
        # a искаме да правим прогноза, обхващаща 
        # numTestPoints периода напред
        npTrainInput = np.array(npScaledData[:-2*self.numTestPoints]).reshape(-1,1)

        # Оформя изхода на групи от по predictions_plus_gap показатели
        npBatchedOutput = np.array([npScaledData[i:i+self.numTestPoints] for i in range(0, self.numAllPoints-self.numTestPoints+1)])
        
        npTrainOutput = np.array(npBatchedOutput[1:-self.numTestPoints])
        
        
        npTestInput = np.array(npScaledData[:-self.numTestPoints]).reshape(-1,1)
       
        
        model = self.constructModel(self.numAllPoints)
        model.fit(x=npTrainInput, y=npTrainOutput, epochs=1000)
        
        rawPredicted = model.predict(npTestInput)
       
        predictedFirst = np.take(rawPredicted, 0, axis=1)

        #print("Predicted_first dimensions:", predicted_first.shape)
        #print(predicted_first)

        predictedLast = np.array(rawPredicted[-1:,1:].reshape(-1))
        #print("Predicted_last dimensions:", predicted_last.shape)
        #print("Predicted_last:", predicted_last)

        predicted = np.concatenate((predictedFirst, predictedLast))

        predicted = predicted.reshape(-1,1)
        predicted = scaler.inverse_transform(predicted)
        predicted = predicted.reshape(-1)
        predicted = np.insert(predicted, 0, np.NaN)
        
        average = (smoothData + predicted.flatten())/2
        
        self.prediction = average
        
        return self.prediction

def getMethod(self):
    return "AveragedRNNExponentialPredictionMethod"

def getParameters(self): 
    params = {}
    params['extended_name']="Average of Double/Triple (for seasonal data) Exponential smoothing and Simple RNN"
    params['alpha']=self.alpha    
    params['beta']=self.beta
    params['gamma']=self.gamma
    return params    
