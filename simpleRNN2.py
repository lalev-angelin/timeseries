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

# Би трябвало да "заковат" генератора на случайни числа така, че
# винаги да произвежда едни и същи последователности от случайни 
# числа. Това на свой ред би трябвало да подобри възпроизводимостта на 
# експериментите. 
seed(1)
tf.random.set_seed(2)

# Спира, ако имаме TensorFlow, който не е построен с поддръжка на  
# GPU или пък няма физически поддържано GPU
assert tf.test.is_built_with_cuda()
assert tf.test.is_gpu_available()

print("******************************************************")
print("* Start ")
print("*")


# Отпечатва броя на наличните физически GPU устройства
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

### ЗАРЕЖДАНЕ НА ДАННИТЕ

data = read_csv('m1.csv', sep=',', decimal=".")
#print(data)

rowno = 2

# Вземаме ред с данни
row = data.iloc[rowno,7:].dropna()
#print("row\n", row)

# datapoints ще съдържа броя на наблюденията
datapoints = data.iloc[rowno,1]
print("datapoints:", datapoints)

# future_predictions ще съдържа броя на периодите в бъдещето, които 
# трябва да прогнозираме
future_predictions = data.iloc[rowno,3]
print("future_predictions:", future_predictions)

# series_name ще съдържа името на серията
series_name = data.iloc[rowno,0]
print("series_name:", series_name)

# series_type ще съдържа типа на серията - годишна, месечна ... 
series_type=data.iloc[rowno,4]
print("series_type", series_type)

# Някои проверки за валидност 
assert datapoints + future_predictions == len(row.values)

 

#### МАЩАБИРАНЕ НА ДАННИТЕ

scaler = MinMaxScaler(feature_range=(0,1))
rownp = row.values.reshape(-1, 1)
rownp = scaler.fit_transform(rownp)
rownp = rownp.reshape(-1)

#print("Rownp:")
#print(rownp)

### СЪЗДАВАНЕ И ОФОРМЯНЕ НА NUMPY МАСИВИТЕ


# Оформя входа на групи от по lookback показатели
inp = rownp.reshape(-1,1)
#print("Input dimensions:\n", inp.shape)
#print("Input:\n", inp)


# Оформя изхода на групи от по predictions_plus_gap показатели
out = np.array([rownp[i:i+future_predictions] for i in range(0, rownp.size-future_predictions+1) ])
#print("Output dimensions:\n", out.shape)
#print("Output:\n", out)

train_inp = inp[:-2*future_predictions]
#print("Train input dimensions\n", train_inp.shape)                
#print("Train input:\n", train_inp)



train_out = out[1:-future_predictions:]
#print("Train output dimensions\n", train_out.shape)
#print("Train output:\n", train_out)


test_inp = inp[:-future_predictions]
#print(test_inp.shape)
#print("Test inpuot: ", test_inp)


#### СЪЗДАВАНЕ И КОМПИЛИРАНЕ НА МОДЕЛА 

model = Sequential()
model.add(SimpleRNN(len(test_inp), input_shape=(1,1), activation='tanh'))
model.add(Dense(units=future_predictions, activation='tanh'))
model.compile(loss='mean_squared_error', optimizer='adam')

#wx = model.get_weights()[0]
#wh = model.get_weights()[1]
#bh = model.get_weights()[2]
#wy = model.get_weights()[3]
#by = model.get_weights()[4]

#print('wx = ', wx, ' wh = ', wh, ' bh = ', bh, ' wy =', wy, 'by = ', by)
#model.summary()
#tf.keras.utils.plot_model (model, show_shapes = True, show_layer_names = True)

model.fit(x=train_inp, y=train_out, epochs=1000)


predicted = model.predict(test_inp)
#print("Predicted dimensions:", predicted.shape)
#print("Predicted:\n", predicted) 


predicted_first = np.take(predicted, 0, axis=1)

#print("Predicted_first dimensions:", predicted_first.shape)
#print(predicted_first)

predicted_last = np.array(predicted[-1:,1:].reshape(-1))
#print("Predicted_last dimensions:", predicted_last.shape)
#print("Predicted_last:", predicted_last)

final_output = np.concatenate((predicted_first, predicted_last))

final_output = final_output.reshape(-1,1)
final_output = scaler.inverse_transform(final_output)
final_output = final_output.reshape(-1)

rownp = rownp.reshape(-1,1)
rownp = scaler.inverse_transform(rownp)
rownp = rownp.reshape(-1)

plt.grid(True, dashes=(1,1))
plt.axvline(x=(rownp.shape[0]-6), color="red", linestyle="--")
plt.title(data.iloc[2,0])
plt.plot(rownp, color='blue')
plt.plot(np.arange(1, len(rownp)), final_output, color='green')
plt.grid(True, dashes=(1,1))
plt.axvline(x=(rownp.shape[0]-6), color="red", linestyle="--")

