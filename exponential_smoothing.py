#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 17:35:01 2023

@author: ownjo
"""

from pandas import read_csv
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from numpy.random import seed
import tensorflow as tf
import sys


### Процедури ###

### Пресмятане на движеща се средна 
### a - Numpy масив
### n - размер на прозореца
def moving_average(input, window):
   ret = np.zeros(length(input)-n)
   for i in (0, length(a)-n-1):
       sum = 0;
       for j in (0, n):
           sum = sum + n[i+j] 
       ret[]
           
   return ret


### Начало на програмата ### 

print("******************************************************")
print("* Start ")
print("*")


### ЗАРЕЖДАНЕ НА ДАННИТЕ

data = read_csv('m1.csv', sep=',', decimal=".")
#print(data)

# Вземаме ред с данни
row = data.iloc[2,7:].dropna()
#print("row\n", row)

# datapoints ще съдържа броя на наблюденията
datapoints = data.iloc[2,1]
print("datapoints:", datapoints)

# future_predictions ще съдържа броя на периодите в бъдещето, които 
# трябва да прогнозираме
future_predictions = data.iloc[2,3]
print("future_predictions:", future_predictions)

# lookback - брой на периодите, които ще гледаме назад, за да направим 
# предсказание
lookback = 6
print("lookback:",lookback)

# series_name ще съдържа името на серията
series_name = data.iloc[2,0]
print("series_name:", series_name)

# series_type ще съдържа типа на серията - годишна, месечна ... 
series_type=data.iloc[2,4]
print("series_type", series_type)

# Някои проверки за валидност 
assert datapoints + future_predictions == len(row.values)

# Начертава оригиналните данни

plt.grid(True, dashes=(1,1))
plt.axvline(x=datapoints, color="red", linestyle="--", label="Prediction horizon")
plt.plot(row, label="Original data")
plt.xticks(rotation=90)
plt.title(series_name+" "+series_type)



np_row = row.values.reshape(-1)
x = np.arange(2, len(np_row))
averages = moving_average(np_row, 3)
plt.plot(x, averages, label="Moving Average (3)")

averages = moving_average(np_row, 2)
x = np.arange(1, len(np_row))
plt.plot(x, averages, label="Moving Average (2)")

averages = moving_average(np_row, 7)
x = np.arange(6, len(np_row))
plt.plot(x, averages, label="Moving Average (7)")

plt.legend()

sys.exit(0)





print(row.iloc[3])

print(scaled)

xvals = np.arange((scaled.shape[0]))
print(xvals)
arr = np.reshape(arr,  (-1,1))

print(xvals)
print(arr)

model.fit(x=xvals, y=scaled, epochs=100)
predicted = model.predict(xvals)


plt.plot(scaled, color='blue')
plt.plot(predicted, color='green', x2=3)
plt.grid(True, dashes=(1,1))
plt.axvline(x=(arr.shape[0]-6), color="red", linestyle="--")
