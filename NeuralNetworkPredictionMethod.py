#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 19:48:27 2023

@author: ownjo
"""
import os
from PredictionMethod import PredictionMethod

class NeuralNetworkPredictionMethod(PredictionMethod):
    
    def __init__(self, data, numTrainPoints, numSeasons=1, title="Neural Network Prediction Method"):
        super().__init__(data,numTrainPoints, numSeasons)
        self.type="nnetpredictionmethod"
        self.title=title
    
    def save(self, directory):
        os.mkdir(directory)
        self.model.save(directory+'/model.keras')
        str.super().toJSON()
        super().save(directory+"/description.csv")