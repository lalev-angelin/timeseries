# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 11:39:14 2023

@author: a.lalev
"""

import os 
from matplotlib import pyplot as plt
import pandas as pd
import json 


def compileStats(filename): 
    print("Compiling stats") 

    files = ["mav.json", "hw.json", "srnn.json", "slstm.json", "arnn.json", "detrended_rnn.json", "boxcox_rnn.json" ]
    descriptions = ["SMA(3)", "HW", "sRNN", "sLSTM", "aRNN", "dRNN", "bcRNN"]

    assert len(files)==len(descriptions)
    
    
    
    resultData = pd.DataFrame()
    row = 1 
    
    for directory in os.listdir():
        if not os.path.isdir(directory):
            continue
        
        print("Processing "+ directory)

        smallestMAPE = 2
        smallestwMAPE = 2
        smallestMAPEDescription = ""
        smallestwMAPEDescription = ""
        
        for i in range(0, len(files)):
            
            resultData.at[row, 'name']=directory
            
            try: 
                f = open(directory+"/"+files[i])
                data = json.load(f)
                f.close()
            except OSError as e:
                continue
        
                
            mape = data['MAPE']
            wmape = data['wMAPE']
            resultData[row, descriptions[i]+"-MAPE"]=mape
            resultData[row, descriptions[i]+"-WMAPE"]=wmape
            
            if (smallestMAPE>mape): 
                smallestMAPE=mape
                smallestMAPEDescription = descriptions[i]
                
            if (smallestwMAPE>wmape):
                smallestwMAPE=wmape
                smallestwMAPEDescription = descriptions[i]
         
        resultData[row, 'minMAPE']=smallestMAPEDescription 
        resultData[row, 'minwMAPE']=smallestwMAPEDescription
        row = row + 1
        
    resultData.to_csv(filename)
    
    
compileStats("stats1.csv")
