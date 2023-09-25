# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 11:39:14 2023

@author: a.lalev
"""

import os 
from matplotlib import pyplot as plt
import json 


def compileStats(filename): 
    print("Compiling stats") 

    i=1
    plotMav = True
    plotHoltWinters = True
    plotSimpleRNN = True
    plotSimpleLSTM = True
    plotCombinedRNN = True
    plotAveragedRNN = True
    plotDetrendedRNN = True
    plotBoxCoxRNN = True
    
    for directory in os.listdir():
        if not os.path.isdir(directory):
            continue
        
        print("Processing "+ directory)

        # Вземаме първия файл и изваждаме оригиналните данни        
        f = open(directory+"/mav.json")
        data = json.load(f)
        f.close()
    
        fig = plt.figure(i, figsize=(28,15))
        i = i + 1
        plt.grid(True, dashes=(1,1))
        plt.title(directory)
        plt.xticks(rotation=90)
        plt.plot(data['data'], color="blue", label="Original data")
        plt.axvline(x=data['numTrainPoints'], color="red", linestyle="--", label="Forecast horizon")
        plt.legend()
        
       
        # Движеща се средна 
        if (plotMav):
            #f = open(directory+"/mav.json")
            #data = json.load(f)
            #f.close()
            plt.plot(data['prediction'], color="orange", label="SMA(3)")
            plt.legend()

        # Двойно или тройно експоненциално изглаждане (зависи дали има данни за сезонност)
        if (plotHoltWinters):
            f = open(directory+"/hw.json")
            data = json.load(f)
            f.close()
            alpha = data['parameters']['smooth_alpha']
            beta = data['parameters']['smooth_beta']
            gamma = data['parameters']['smooth_gamma']
            plt.plot(data['prediction'], color="red", label="Holt-Winters $\\alpha\\approx$%s, $\\beta\\approx$%s $\\gamma\\approx$%s"% 
                     (round(alpha,4), round(beta,4), round(gamma,4)))
            plt.legend()
            
        # Проста RNN
        if (plotSimpleRNN):  
            f = open(directory+"/srnn.json")
            data = json.load(f)
            f.close()
            plt.plot(data['prediction'], color="green", label="Simple RNN")
            plt.legend()
            
        # Проста LSTM     
        if (plotSimpleLSTM):  
            f = open(directory+"/slstm.json")
            data = json.load(f)
            f.close()
            plt.plot(data['prediction'], color="brown", label="Simple LSTM")
            plt.legend()
        
        # Комбиниран RNN
        if (plotCombinedRNN):
            f = open(directory+"/crnn.json")
            data = json.load(f)
            f.close()
            plt.plot(data['prediction'], color="darkgray", label="Combined RNN")
            plt.legend()

        # Усреднен RNN
        if (plotAveragedRNN):
            f = open(directory+"/arnn.json")
            data = json.load(f)
            f.close()
            plt.plot(data['prediction'], color="yellow", label="Averaged RNN")
            plt.legend()

        # Детренднат RNN
        if (plotDetrendedRNN):
            f = open(directory+"/detrended_rnn.json")
            data = json.load(f)
            f.close()
            plt.plot(data['prediction'], color="green", linestyle="--", label="Linear Detrend + RNN")
            plt.legend()
            
        # BoxCox + RNN
        if (plotBoxCoxRNN):
            f = open(directory+"/boxcox_rnn.json")
            data = json.load(f)
            f.close()
            plt.plot(data['prediction'], color="green", linestyle="-.", label="BoxCox + RNN")
            plt.legend()
                        
            
        plt.savefig(directory+"/"+filename)

          
compileStats("fig1.png")
