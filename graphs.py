#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 22:15:25 2023

@author: ownjo
"""
from pandas import read_csv
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from MovingAveragePredictionMethod import MovingAveragePredictionMethod
from ExponentialPredictionMethod import ExponentialPredictionMethod

### ЗАРЕЖДАНЕ НА ДАННИТЕ

data = read_csv('m1.csv', sep=',', decimal=".")
#print(data)

pdf = PdfPages("graphs_m1.pdf")

for q in range (0, len (data)):

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
    
    # Движеща се средна
    average = MovingAveragePredictionMethod(row, datapoints, window=3)
    prediction = average.predict()
        
    
    fig = plt.figure(q, figsize=(16,9))
    plt.grid(True, dashes=(1,1))
    plt.title(series_name+series_type)
    plt.xticks(rotation=90)
    plt.plot(average.data, color="blue", label="Original data")
    plt.axvline(x=datapoints, color="red", linestyle="--", label="Forecast horizon")
    plt.plot(prediction, color="orange", label="Moving average 3, MAPE=%s, wMAPE=%s" % (average.computeMAPE(), average.computeWMAPE()))
    plt.legend()

    # Просто експоненциално изглаждане
    exponential = ExponentialPredictionMethod(row, datapoints)
    prediction = exponential.predict()
    plt.plot(prediction, color="magenta", label="Exp. smoothing $\\alpha=%s$, MAPE=%s, wMAPE=%s" % 
              (exponential.hwresults.params['smoothing_level'], exponential.computeMAPE(), exponential.computeWMAPE()))
    plt.legend()


    # Двойно експоненциално изглаждане

    pdf.savefig(fig)
    
pdf.close()
