#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 20:56:13 2023

@author: ownjo
"""

from PredictionMethod import PredictionMethod
from MovingAveragePredictionMethod import MovingAveragePredictionMethod
from DummyPredictionMethod import DummyPredictionMethod
import unittest
import cmath

class Test(unittest.TestCase): 
    def test_PredictionMethod(self): 
        data=[0,1,2,3,4,5,6,7,8,9]
        prediction = PredictionMethod(data, 5)
        self.assertEqual(prediction.data, data)
        self.assertEqual(prediction.trainData,  [0,1,2,3,4])
        self.assertEqual(prediction.testData, [5,6,7,8,9])
        self.assertEqual(prediction.numAllPoints, 10)
        self.assertEqual(prediction.numTrainPoints, 5)
        self.assertEqual(prediction.numTestPoints, 5)
        
    def test_MovingAveragePredictionMethod(self):
        data=[0,1,2,3,4,5,6,7,8,9]
        prediction = MovingAveragePredictionMethod(data, 9, window=3)
        prediction.predict()
        self.assertEqual(prediction.prediction, [cmath.nan, cmath.nan, cmath.nan, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.0])

    def test_Mape(self):
        data=[0,1,2,3,4,5,6,7,8,9]
        prediction = MovingAveragePredictionMethod(data, 7)
        pred = prediction.predict()
        self.assertEqual(prediction.computeMAPE(), 0.5906819517930629)
        self.assertEqual(prediction.computeWMAPE(), [nan, nan, nan, 1.0, 2.0, 3.0, 4.0, 3.0, 3.3333333333333335, 3.4444444444444446])
        
    def test_PredictionMethodSave(self):
        data=[0,1,2,3,4,5,6,7,8,9]
        prediction = MovingAveragePredictionMethod(data, 7)
        pred = prediction.predict()
        prediction.save('test.json')
        
    def test_DummyPredictionMethod(self):
        data=[0,1,2,3,4,5,6,7,8,9]
        prediction = DummyPredictionMethod(data, 3)
        self.assertEqual(prediction, [0,1,2,3,4,5,6,7,8,9])
        
        
if __name__ == '__main__':
    unittest.main()
