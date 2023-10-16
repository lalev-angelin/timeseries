# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 20:16:09 2023

@author: Owner
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import math
from MovingAveragePredictionMethod import MovingAveragePredictionMethod
from DoubleExponentialPredictionMethod import DoubleExponentialPredictionMethod
from TripleExponentialPredictionMethod import TripleExponentialPredictionMethod
from SimpleRNNEnsemblePredictionMethod import SimpleRNNEnsemblePredictionMethod
import sys 
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM, Dropout

class SpecialRNNEnsemblePredictionMethod(SimpleRNNEnsemblePredictionMethod):
    
    def constructModel(self, period):
       
        if self.numNeurons[0] == None: 
            for q in range(0, self.numTestPoints):
               rnn1NeuronCount = round(self.numAllPoints / 2)
               model = Sequential()
               model.add(SimpleRNN(rnn1NeuronCount, input_shape=(self.window,1), activation='tanh', return_sequences=False))
               model.add(Dense(units=1, activation='tanh'))
               model.compile(loss='mean_squared_error', optimizer='adam')
           
               self.layer_description={}
               self.layer_description['model' + str(period) + 'layer1_type']="SimpleRNN"
               self.layer_description['model' + str(period) + 'layer1_neuron_count']=rnn1NeuronCount
               self.layer_description['model' + str(period) + 'layer1_activation']="tanh"
               self.layer_description['model' + str(period) + 'layer2_type']="Dense"
               self.layer_description['model' + str(period) + 'layer2_neuron_count']=self.numTestPoints
               self.layer_description['model' + str(period) + 'layer2_activation']="tanh"
               self.layer_description['model' + str(period) + 'loss']="mean_squared_error"
               self.layer_description['model' + str(period) + 'optimizer']='adam'
               return model
           
        else:
           sys.exit(1)
           
        return None


rawx = np.arange(0, 10*math.pi, step = 0.5)
rawy = np.sin(rawx)

raw2y = np.multiply(rawx, 0.5)

future_predictions = round(len(rawy)/4)
datapoints = len(rawx) - future_predictions
seasonality = 5 # 5 * 2pi seasons
row = rawy+raw2y

print("Datapoints: "+str(datapoints))
print("Future predictions: "+str(future_predictions))
 
average = MovingAveragePredictionMethod(row, datapoints, window=3)
prediction = average.predict()
    

fig = plt.figure(1, figsize=(28,15))
plt.grid(True, dashes=(1,1))
plt.title("Sin")
plt.xticks(rotation=90)
plt.plot(average.data, color="blue", label="Original data")
plt.axvline(x=datapoints, color="red", linestyle="--", label="Forecast horizon")
plt.plot(prediction, color="orange", label="Moving average 3, MAPE=%s, wMAPE=%s" % (average.computeMAPE(), average.computeWMAPE()))
plt.legend()

# Тройно експоненциално изглаждане
if seasonality>1:
    holtwinters = TripleExponentialPredictionMethod(row, datapoints, numSeasons=seasonality)
else:
    holtwinters = DoubleExponentialPredictionMethod(row, datapoints)
prediction = holtwinters.predict()
plt.plot(prediction, color="red", label="Exp. smoothing $\\alpha=%s$, $\\beta=%s$ $\\gamma=%s$, MAPE=%s, wMAPE=%s" % 
          (holtwinters.hwresults.params['smoothing_level'], holtwinters.hwresults.params['smoothing_trend'], 
           holtwinters.hwresults.params['smoothing_seasonal'], holtwinters.computeMAPE(), holtwinters.computeWMAPE()))
plt.legend()

# Special RNN Ensemble
simpleRNNEnsemble = SpecialRNNEnsemblePredictionMethod(row, datapoints, window=3)
prediction = simpleRNNEnsemble.predict()
plt.plot(prediction, color="green", label="Simple RNN Ensemble, MAPE=%s, wMAPE=%s" % (simpleRNNEnsemble.computeMAPE(), simpleRNNEnsemble.computeWMAPE()))
plt.legend()
