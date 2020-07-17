# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 18:47:04 2020

@author: Medha
"""

 
from flask import Flask,request,render_template
import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt 
import pickle
from sklearn.preprocessing import PolynomialFeatures


app = Flask(__name__)
model = pickle.load(open('modelPoly.pkl','rb')) 

@app.route('/')
def home():
    return render_template('flightsHome.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
        For rendering results on HTML GUI
    '''
    
    WHEELS_OFF = request.form['WHEELS_OFF']
    SCHEDULED_TIME = request.form['SCHEDULED_TIME']
    ELAPSED_TIME = request.form['ELAPSED_TIME']
    ARRIVAL_DELAY = request.form['ARRIVAL_DELAY']
    AIRLINE_DELAY = request.form['AIRLINE_DELAY']
    LATE_AIRCRAFT_DELAY = request.form['LATE_AIRCRAFT_DELAY']
    WEATHER_DELAY = request.form['WEATHER_DELAY']
    AIR_TIME = request.form['AIR_TIME']
    TAXI_IN = request.form['TAXI_IN']
 
    data = [{'WHEELS_OFF': WHEELS_OFF, 'SCHEDULED_TIME': SCHEDULED_TIME, 
             'ELAPSED_TIME':ELAPSED_TIME, 'AIR_TIME':AIR_TIME,
             'TAXI_IN':TAXI_IN, 'ARRIVAL_DELAY':ARRIVAL_DELAY,
             'AIRLINE_DELAY':AIRLINE_DELAY, 'AIR_TIME':AIR_TIME,
             'LATE_AIRCRAFT_DELAY':LATE_AIRCRAFT_DELAY, 'WEATHER_DELAY':WEATHER_DELAY
             }] 
    X_Test = pd.DataFrame(data) 
    

    # PolynomialFeatures (prepreprocessing)
    poly = PolynomialFeatures(degree=3)
    testPoly = poly.fit_transform(X_Test)
    #    model = pickle.load(open('model.pkl','rb'))
    predicted= model.predict(testPoly)
    
##    int_features = request.get('review')
#    if predicted == 'No':
#       output = 1    
#    elif predicted == 'Yes':
#       output = 2

#       
    parametersText = 'abcd'
#    return  render_template('deployMed.html',prediction_text = 'Drug Review Rating contains $ {}'.format(output))
    return render_template('flightsResult.html',prediction = predicted,textOutput = parametersText)
 
if __name__ == '__main__':
	app.run(debug=True)