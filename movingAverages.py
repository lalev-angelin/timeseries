#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 17:35:01 2023

@author: ownjo
"""

from pandas import read_csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from numpy.random import seed
import sys

#####################################################
# Изчислява движеща се средна от  numpy масив nparr, 
# като използва прозорец с размер, зададен от window
# 
def movingAverage(nparr, window):
    # Масивът, в който ще върнем изчислените 
    # стойности
    ret = np.zeros(nparr.size-window+1)

   
    for i in range(0, nparr.size-window+1):
        masum = 0
        for j in range(i, i+window):
            masum = masum + nparr[j]
#        print(i, masum)
        ret[i]=masum/window
        
    return ret

####################################################
# Метод със същото предназначение, взет от Интернет
# 
def movingAverage1(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


#####################################################
# 
def movingAverageWithPad(data, window, 
                         num_predictions):

    ret = movingAverage(data, window)
    print(data)
    
    source_position = len(data)
    padded_data = np.pad(data, (0, num_predictions)).astype('double')
    
    
    remaining_predictions = num_predictions-1

    start_position = len(ret)
    ret = np.pad(ret, (0, remaining_predictions))
    stop_position = len(ret)
    
    #print(padded_data)
    #print(ret)
    
    #print(source_position)
    #print(start_position)
    #print(stop_position)

    padded_data[source_position]=ret[start_position-1]
    for c in range(start_position, stop_position):
#        print(padded_data)
        part = padded_data[source_position-window+1: source_position+1]
#        print(part)
        source_position=source_position+1
        padded_data[source_position]=movingAverage(part, window)[0]
        ret[c]=padded_data[source_position]
#        print(padded_data)
    return ret

        
   
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

# series_name ще съдържа името на серията
series_name = data.iloc[2,0]
print("series_name:", series_name)

# series_type ще съдържа типа на серията - годишна, месечна ... 
series_type=data.iloc[2,4]
print("series_type", series_type)

# Някои проверки за валидност 
assert datapoints + future_predictions == len(row.values)

# Оригиналните данни в numpy масив
original_data = row.values

# Данните, които могат да се ползват
# за обучение на модела
train_data = original_data[:-future_predictions]

# Големина на прозореца на движещата се средна
window = 3


average = movingAverageWithPad(train_data, window, future_predictions)

#plt.figure(figsize=(16,9))
plt.grid(True, dashes=(1,1))
plt.title(series_name+series_type)
plt.xticks(rotation=90)
plt.plot(original_data, color="blue", label="Original data")
plt.axvline(x=datapoints, color="red", linestyle="--", label="Forecast horizon")
plt.plot(np.arange(window, len(average)+window), average, color="orange", label="Moving average 3")
plt.legend()
sys.exit(0)

moving_average = movingAverage(original_data, 3)

