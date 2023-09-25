# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 11:39:14 2023

@author: a.lalev
"""

import os 
from matplotlib import pyplot as plt
import json 

files = ["mav.json", "hw.json"]
colors = ["cyan", "green"]
labels = ["Original "

def compileStats(): 
    print("Compiling stats") 
    
    for directory in os.listdir():
        if not os.path.isdir(directory):
            continue
        for i in range(0, len(files): 
            f = open(directory+"/"+files[i])
            data = json.load(f)
            f.close()
        
            fig = plt.figure(data['data'], figsize=(28,15))
            plt.grid(True, dashes=(1,1))
            plt.title(directory)
            plt.xticks(rotation=90)
            plt.plot(average.data, color=colors[i], label="Original data")
            plt.axvline(x=datapoints, color="red", linestyle="--", label="Forecast horizon")
            plt.plot(prediction, color="orange", label="Moving average 3, MAPE=%s, wMAPE=%s" % (average.computeMAPE(), average.computeWMAPE()))
            plt.legend()
        


compileStats("fig1.png")
