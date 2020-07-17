# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 18:00:13 2020

@author: Medha
"""

#from flask import Flask,request,render_template
import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt 
import pickle

model = pickle.load(open('modelPoly.pkl','rb')) 

WHEELS_OFF = 1615
SCHEDULED_TIME = 97
ELAPSED_TIME = 100
ARRIVAL_DELAY = 26
AIRLINE_DELAY = 13
LATE_AIRCRAFT_DELAY = 0
WEATHER_DELAY = 0
AIR_TIME = 82
TAXI_IN = 9
 
data = [{'WHEELS_OFF': WHEELS_OFF, 'SCHEDULED_TIME': SCHEDULED_TIME, 
             'ELAPSED_TIME':ELAPSED_TIME, 'AIR_TIME':AIR_TIME,
             'TAXI_IN':TAXI_IN, 'ARRIVAL_DELAY':ARRIVAL_DELAY,
             'AIRLINE_DELAY':AIRLINE_DELAY, 'AIR_TIME':AIR_TIME,
             'LATE_AIRCRAFT_DELAY':LATE_AIRCRAFT_DELAY, 'WEATHER_DELAY':WEATHER_DELAY
             }] 
X_Test = pd.DataFrame(data) 
    
print(X_Test.shape)
print( X_Test['WHEELS_OFF'])
print( X_Test['SCHEDULED_TIME'])
print(  X_Test['ELAPSED_TIME'])
print(  X_Test['ARRIVAL_DELAY'])
print(  X_Test['AIRLINE_DELAY'])
print( X_Test['LATE_AIRCRAFT_DELAY'])
print( X_Test['WEATHER_DELAY'])
print(  X_Test['AIR_TIME'])
print(  X_Test['TAXI_IN'])
    

from sklearn.preprocessing import PolynomialFeatures
#from sklearn.linear_model import LinearRegression

# PolynomialFeatures (prepreprocessing)
poly = PolynomialFeatures(degree=3)
testPoly = poly.fit_transform(X_Test)
#    model = pickle.load(open('model.pkl','rb'))
predicted= model.predict(testPoly)
    