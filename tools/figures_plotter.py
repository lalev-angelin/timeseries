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


        if (plotHoltWinters):
            f = open(directory+"/hw.json")
            data = json.load(f)
            f.close()
            alpha = data['params']['alpha']
            beta = data['params']['beta']
            gamma = data['params']['gamma']
            plt.plot(data['prediction'], color="red", label="Holt-Winters $\\alpha=%s$, $\\beta=%s$ $\\gamma=%s$"% 
                     (alpha, beta, gamma))
            plt.legend()
            
        
        plt.savefig(directory+"/"+filename)

          
compileStats("fig1.png")
