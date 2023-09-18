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
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from matplotlib.backends.backend_pdf import PdfPages
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
import tensorflow as tf



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
    #print(data)
    
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

pdf = PdfPages("graphs_m1.pdf")

#for q in range (0, len(data)):
for q in range(250, 290):
    # Номер на ред с данни
    rowno = q
    
    # Вземаме ред с данни
    row = data.iloc[rowno,7:].dropna()
    print("row\n", row)
    
    
    # datapoints ще съдържа броя на наблюденията
    datapoints = data.iloc[rowno,1]
    print("datapoints:", datapoints)
    
    # Брой сезони, 1 - ако няма сезонност
    seasonality = data.iloc[rowno, 2]
    
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
    
    # Оригиналните данни в numpy масив
    original_data = row.values
    
    # Данните, които могат да се ползват
    # за обучение на модела
    train_data = original_data[:-future_predictions]
    
    # Големина на прозореца на движещата се средна
    window = 3
    
    
    average = movingAverageWithPad(train_data, window, future_predictions)
    
    fig = plt.figure(q, figsize=(16,9))
    plt.grid(True, dashes=(1,1))
    plt.title(series_name+series_type)
    plt.xticks(rotation=90)
    plt.plot(original_data, color="blue", label="Original data")
    plt.axvline(x=datapoints, color="red", linestyle="--", label="Forecast horizon")
    plt.plot(np.arange(window, len(average)+window), average, color="orange", label="Moving average 3")
    plt.legend()
    
    
    # Това са три класа. SimpleExpSmoothing прави просто експоненциално 
    # изглаждане. Holt прави двойно експоненциално изглаждане. 
    # 
    
    # Просто
    
    hwresults = SimpleExpSmoothing(train_data, initialization_method='estimated').fit()
    forecast = hwresults.predict(start=0, end=len(train_data)+future_predictions-1)
    
    #print(hwresults.params)
    #sys.exit(1)
    
    plt.plot(forecast, color="green", 
             label="SES (est $\\alpha=%s)$" % round(hwresults.params['smoothing_level'], 20))
    plt.legend()
    
    # Холт 
    
    # Забележете, че ако изпуснем параметрите на fit 
    # smoothing_level= и smoothing_trend=,
    # то си ги оптимизира. Не знаем точно как, но видях max-log-likehood някъде
    # в документацията?
     
    hwresults = Holt(train_data, initialization_method='estimated').fit(optimized=True)
    forecast = hwresults.predict(start=0, end=len(train_data)+future_predictions-1)
    plt.plot(forecast, color="black", 
             label="Holt $\\alpha=%s$, $\\beta=%s$" % 
             (round(hwresults.params['smoothing_level'],20), 
                   round(hwresults.params['smoothing_trend'],20)))
    plt.legend()
    
    
    if seasonality>1:
    
        # Холт - Уинтърс
        hwresults = ExponentialSmoothing(train_data, 
                                         initialization_method='estimated', trend="add", 
                                         seasonal="add", seasonal_periods=seasonality).fit(optimized=True)
        forecast = hwresults.predict(start=0, end=len(train_data)+future_predictions-1)
        
        plt.plot(forecast, color="magenta", 
                 label="Holt-Winters $\\alpha=%s$, $\\beta=%s$, $\\gamma=%s$" % 
                 (round(hwresults.params['smoothing_level'],20), 
                       round(hwresults.params['smoothing_trend'],20), 
                       round(hwresults.params['smoothing_seasonal'], 20)))
        plt.legend()




    #### RNN
    
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
    
    plt.plot(np.arange(1, len(rownp)), final_output, color='brown', label="RNN Conf 1")
    plt.legend()



    #### RNN2
    
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
    if seasonality>1:
        model.add(SimpleRNN(seasonality, input_shape=(1,1)), activation='tanh')
        model.add(SimpleRNN(len(test_inp), activation='tanh'))
    else:
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
    
    plt.plot(np.arange(1, len(rownp)), final_output, color='cyan', label="RNN Conf 1")
    plt.legend()
    pdf.savefig(fig)
    
pdf.close()


