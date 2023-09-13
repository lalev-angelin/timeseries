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


#### МАЩАБИРАНЕ НА ДАННИТЕ

scaler = MinMaxScaler(feature_range=(0,1))
print(row.values.reshape(-1,29))
sys.exit(0)
row = pd.DataFrame(scaler.fit_transform(row.values.reshape(-1,2)))
print(row) 
sys.exit(1)


#### СЪЗДАВАНЕ И КОМПИЛИРАНЕ НА МОДЕЛА 

model = Sequential()
model.add(SimpleRNN(lookback*6, input_shape=(lookback,1), activation='tanh'))
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


### СЪЗДАВАНЕ И ОФОРМЯНЕ НА NUMPY МАСИВИТЕ

# Създава от данните последователност от вектори, чиито компоненти са 
# стойността в момент t, t-1, t-2 и така lookback периода назад
arr = row.values.astype('float32').flatten()
print("Arr:\n", arr)
print("Arr dimensions:\n", arr.shape)

# Отделя входа
inp = np.array([ arr[i:i+lookback] for i in range(datapoints-lookback-future_predictions+1) ])
print("Input:\n", inp)
print("Input dimensions:\n", inp.shape)

out = np.array([ arr[i:i+future_predictions] for i in range(lookback, datapoints-future_predictions+1) ])
print("Output:\n", out)

test_inp = np.array( arr[len(arr)-future_predictions-lookback : len(arr) - future_predictions])
print("Test input:\n", test_inp)

test_out = np.array( arr[len(arr)-future_predictions : len(arr)] )
print("Test output:\n", test_out)

model.fit(x=inp, y=out, epochs=100)

print(inp.shape)
print(test_inp.shape)

test_inp=np.reshape(test_inp, (1, -1))
print(test_inp)
print(test_inp.shape)

pred_inp = np.concatenate((inp, test_inp), axis=0)
print(pred_inp)
print(pred_inp.shape)

predicted = model.predict(pred_inp)
print("Predicted:\n", predicted) 
                               
final_output = [i[0] for i in predicted]
final_output = np.concatenate((np.zeros(6), final_output, predicted[-1][-5:]), axis=0) 
print(final_output)                                

sys.exit(0)


plt.plot(scaled)
plt.grid(True, dashes=(1,1))
plt.axvline(x=(arr.shape[0]-6), color="red", linestyle="--")

print(row.iloc[3])
plt.title(data.iloc[2,0])

print(scaled)

xvals = np.arange((scaled.shape[0]))
print(xvals)
arr = np.reshape(arr,  (-1,1))

print(xvals)
print(arr)

model.fit(x=xvals, y=scaled, epochs=100)
predicted = model.predict(xvals)


plt.plot(scaled, color='blue')
plt.plot(predicted, color='green')
plt.grid(True, dashes=(1,1))
plt.axvline(x=(arr.shape[0]-6), color="red", linestyle="--")
