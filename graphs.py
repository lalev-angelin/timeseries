#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 22:15:25 2023

@author: ownjo
"""
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from MovingAveragePredictionMethod import MovingAveragePredictionMethod
from ExponentialPredictionMethod import ExponentialPredictionMethod
from DoubleExponentialPredictionMethod import DoubleExponentialPredictionMethod
from TripleExponentialPredictionMethod import TripleExponentialPredictionMethod
from SimpleRNNPredictionMethod import SimpleRNNPredictionMethod
from SimpleLSTMPredictionMethod import SimpleLSTMPredictionMethod
from CombinedRNNPredictionMethod import CombinedRNNPredictionMethod
from AveragedRNNExponentialPredictionMethod import AveragedRNNExponentialPredictionMethod
from FeedForwardPredictionMethod import FeedForwardPredictionMethod
from DetrendingDecorator import DetrendingDecorator
from BoxCoxDecorator import BoxCoxDecorator

import pandas as pd
import os

### ЗАРЕЖДАНЕ НА ДАННИТЕ

data = pd.read_csv('m1.csv', sep=',', decimal=".")

pdf = PdfPages("results/graphs.pdf")

resultData = pd.DataFrame()

#for q in range (0, len (data)):
#for q in range (260, 270):
for q in range (0, 10):

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
    series_name = data.iloc[rowno,0].lstrip().rstrip()
    print("series_name:", series_name)
    
    # series_type ще съдържа типа на серията - годишна, месечна ... 
    series_type=data.iloc[rowno,4]
    print("series_type", series_type)
    
    # Някои проверки за валидност 
    assert datapoints + future_predictions == len(row.values)
    

    # Проверяваме дали вече не сме го правили 
    if os.path.isdir("results/"+series_name):
        continue
    else:
        os.mkdir("results/"+series_name)

   
    resultData.at[q, 'name']=series_name
    resultData.at[q,'datapoints']=datapoints
    resultData.at[q, 'future_predictions']=future_predictions
   
    # Движеща се средна
    average = MovingAveragePredictionMethod(row, datapoints, window=3)
    prediction = average.predict()
        
    
    fig = plt.figure(q, figsize=(28,15))
    plt.grid(True, dashes=(1,1))
    plt.title(series_name+series_type)
    plt.xticks(rotation=90)
    plt.plot(average.data, color="blue", label="Original data")
    plt.axvline(x=datapoints, color="red", linestyle="--", label="Forecast horizon")
    plt.plot(prediction, color="orange", label="Moving average 3, MAPE=%s, wMAPE=%s" % (average.computeMAPE(), average.computeWMAPE()))
    plt.legend()

    resultData.at[q, 'ma_mape']=average.computeMAPE()
    resultData.at[q, 'ma_wmape']=average.computeWMAPE()

    average.save("results/"+series_name+"/mav.json")

    # # Просто експоненциално изглаждане
    # exponential = ExponentialPredictionMethod(row, datapoints)
    # prediction = exponential.predict()
    # plt.plot(prediction, color="magenta", label="Exp. smoothing $\\alpha=%s$, MAPE=%s, wMAPE=%s" % 
    #           (exponential.hwresults.params['smoothing_level'], exponential.computeMAPE(), exponential.computeWMAPE()))
    # plt.legend()

    # resultData.at[q, 'es_mape']=exponential.computeMAPE()
    # resultData.at[q, 'es_wmape']=exponential.computeWMAPE()

    
    # # Двойно експоненциално изглаждане
    # holt = DoubleExponentialPredictionMethod(row, datapoints)
    # prediction = holt.predict()
    # plt.plot(prediction, color="black", label="Exp. smoothing $\\alpha=%s$, $\\beta=%s$, MAPE=%s, wMAPE=%s" % 
    #           (holt.hwresults.params['smoothing_level'], holt.hwresults.params['smoothing_trend'], holt.computeMAPE(), holt.computeWMAPE()))
    # plt.legend()

    # resultData.at[q, 'holt_mape']=holt.computeMAPE()
    # resultData.at[q, 'holt_wmape']=holt.computeWMAPE()

    # Тройно експоненциално изглаждане
    if seasonality>1:
        holtwinters = TripleExponentialPredictionMethod(row, datapoints, numSeasons=seasonality)
    else:
        holtwinters = DoubleExponentialPredictionMethod(row, datapoints)
    prediction = holtwinters.predict()
    holtwinters.save("results/"+series_name+"/hw.json")
    plt.plot(prediction, color="red", label="Exp. smoothing $\\alpha=%s$, $\\beta=%s$ $\\gamma=%s$, MAPE=%s, wMAPE=%s" % 
              (holtwinters.hwresults.params['smoothing_level'], holtwinters.hwresults.params['smoothing_trend'], 
               holtwinters.hwresults.params['smoothing_seasonal'], holtwinters.computeMAPE(), holtwinters.computeWMAPE()))
    plt.legend()

    resultData.at[q, 'holtwinters_mape']=holtwinters.computeMAPE()
    resultData.at[q, 'holtwinters_wmape']=holtwinters.computeWMAPE()


    # Проста RNN
    simpleRNN = SimpleRNNPredictionMethod(row, datapoints)
    prediction = simpleRNN.predict()
    simpleRNN.save("results/"+series_name+"/srnn.json")
    plt.plot(prediction, color="cyan", label="Simple RNN, MAPE=%s, wMAPE=%s" % (simpleRNN.computeMAPE(), simpleRNN.computeWMAPE()))
    plt.legend()
    
    resultData.at[q, 'simplernn_mape']=simpleRNN.computeMAPE()
    resultData.at[q, 'simplernn_wmape']=simpleRNN.computeWMAPE()
    
    # Проста LSTM
    simpleLSTM = SimpleLSTMPredictionMethod(row, datapoints)
    prediction = simpleLSTM.predict()
    simpleLSTM.save("results/"+series_name+"/slstm.json")
    plt.plot(prediction, color="brown", label="Simple LSTM, MAPE=%s, wMAPE=%s" % (simpleLSTM.computeMAPE(), simpleLSTM.computeWMAPE()))
    plt.legend()
    
    resultData.at[q, 'simplelstm_mape']=simpleLSTM.computeMAPE()
    resultData.at[q, 'simplelstm_wmape']=simpleLSTM.computeWMAPE()

    # Комбиниран RNN
    combinedRNN = CombinedRNNPredictionMethod(row, datapoints, numSeasons=seasonality)
    prediction = combinedRNN.predict()
    simpleLSTM.save("results/"+series_name+"/crnn.json")
    plt.plot(prediction, color="darkgray", label="Combined RNN, MAPE=%s, wMAPE=%s" % (combinedRNN.computeMAPE(), combinedRNN.computeWMAPE()))
    plt.legend()
    
    resultData.at[q, 'combinedrnn_mape']=combinedRNN.computeMAPE()
    resultData.at[q, 'combinedrnn_wmape']=combinedRNN.computeWMAPE()    

    # Усреднен RNN
    averagedRNN = AveragedRNNExponentialPredictionMethod(row, datapoints, numSeasons=seasonality)
    prediction = averagedRNN.predict()
    averagedRNN.save("results/"+series_name+"/arnn.json")
    plt.plot(prediction, color="yellow", label="Averaged RNN, MAPE=%s, wMAPE=%s" % (averagedRNN.computeMAPE(), averagedRNN.computeWMAPE()))
    plt.legend()
    
    resultData.at[q, 'averaged_mape']=averagedRNN.computeMAPE()
    resultData.at[q, 'averaged_wmape']=averagedRNN.computeWMAPE()    

    
    plt.savefig("results/"+series_name+"/fig.png")
    pdf.savefig(fig)
    
    # RNN с детрендинг
    detrendedRNN = DetrendingDecorator(SimpleRNNPredictionMethod(row, datapoints))
    prediction = detrendedRNN.predict()
    detrendedRNN.save("results/"+series_name+"/detrended_rnn.json")
    plt.plot(prediction, color="cyan", linestyle="--", label="Detrended RNN, MAPE=%s, wMAPE=%s" % (detrendedRNN.computeMAPE(), detrendedRNN.computeWMAPE()))
    plt.legend()
    
    
    # RNN с Бокс-Кокс
    boxcoxRNN = BoxCoxDecorator(SimpleRNNPredictionMethod(row, datapoints))
    prediction = boxcoxRNN.predict()
    boxcoxRNN.save("results/"+series_name+"/boxcox_rnn.json")
    plt.plot(prediction, color="cyan", linestyle="-.-", label="BoxCox RNN, MAPE=%s, wMAPE=%s" % (boxcoxRNN.computeMAPE(), boxcoxRNN.computeWMAPE()))
    plt.legend()




pdf.close()
resultData.to_csv('results/data.csv')
