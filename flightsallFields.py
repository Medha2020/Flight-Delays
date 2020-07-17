# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 22:04:58 2020

@author: Medha
"""



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
#from mpl_toolkits.basemap import Basemap
import datetime
from pandas_profiling import ProfileReport
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
import sklearn.metrics as metrics
from sklearn.model_selection import cross_val_score, GridSearchCV
import pickle 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier 
#flights = pd.read_csv("C:\\Users\\Medha\\Desktop\\Medha\\ExcelR\\Flights\\Dataset\\flights_sample.csv")

######Airport : Atlanta Airline : WN
flights = pd.read_csv("C:\\Users\\Medha\\Desktop\\Medha\\ExcelR\\Flights\\Dataset\\filtered_flights.csv")

######Airport : Atlanta 
flights = pd.read_csv("C:\\Users\\Medha\\Desktop\\Medha\\ExcelR\\Flights\\Dataset\\flights_filtered.csv")

######Filghts : 20 % data 
flights = pd.read_csv("C:\\Users\\Medha\\Desktop\\Medha\\ExcelR\\Flights\\Dataset\\sampleFlights.csv")

flights_bk = flights
flights = flights_bk.sample(frac =0.10)
######Airports
airports = pd.read_csv("C:\\Users\\Medha\\Desktop\\Medha\\ExcelR\\Flights\\Dataset\\airports.csv")


#profile = ProfileReport(flights, title="Pandas Profiling Report")
#profile.to_file('C:\\Users\\Medha\\Desktop\\Medha\\ExcelR\\Flights\\Dataset\\profile_report.html')
flights.shape
flights.columns
flights.drop(flights[flights["CANCELLED"] == 1].index, inplace=True)
flights = flights.drop(['YEAR'], axis = 1) 
flights = flights.drop(['CANCELLATION_REASON'], axis = 1) 
flights = flights.drop(['CANCELLED'], axis = 1) 
flights = flights.drop(['DIVERTED'], axis = 1) 
flights = flights.fillna(0)



flights.head(20)
np.corrcoef(flights.DEPARTURE_DELAY,flights.TAXI_OUT)
np.corrcoef(flights.DEPARTURE_DELAY,flights.WHEELS_OFF)
np.corrcoef(flights.DEPARTURE_DELAY,flights.WHEELS_ON)
np.corrcoef(flights.DEPARTURE_DELAY,flights.TAXI_IN)
np.corrcoef(flights.DEPARTURE_DELAY,flights.AIR_SYSTEM_DELAY)
np.corrcoef(flights.DEPARTURE_DELAY,flights.SECURITY_DELAY)
np.corrcoef(flights.DEPARTURE_DELAY,flights.AIRLINE_DELAY)
np.corrcoef(flights.DEPARTURE_DELAY,flights.WEATHER_DELAY)
np.corrcoef(flights.DEPARTURE_DELAY,flights.LATE_AIRCRAFT_DELAY)

plt.scatter(flights.DEPARTURE_DELAY,flights.AIR_SYSTEM_DELAY,c="r");plt.xlabel("DEPARTURE_DELAY");plt.ylabel("AIR_SYSTEM_DELAY")
plt.boxplot(flights.LATE_AIRCRAFT_DELAY)
plt.hist(flights.WEATHER_DELAY)
plt.bar(flights.DEPARTURE_DELAY,flights.AIR_SYSTEM_DELAY)
plt.show()

plt.bar(flights.AIRLINE,flights.SECURITY_DELAY)
plt.show()

plt.hist(flights.LATE_AIRCRAFT_DELAY)
plt.hist(flights.WEATHER_DELAY)
plt.hist(flights.AIRLINE_DELAY)
plt.hist(flights.TAXI_IN)
plt.hist(flights.SECURITY_DELAY)
plt.hist(flights.AIR_SYSTEM_DELAY)
corMatrix = flights.corr()

############################ Data Cleaning####################################

#flights_model = flights.drop(['MONTH', 'DAY', 'DAY_OF_WEEK', 'AIRLINE', 'FLIGHT_NUMBER',
#       'TAIL_NUMBER', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT',
#       'SCHEDULED_DEPARTURE', 'DEPARTURE_TIME', 'TAXI_OUT',
#       'WHEELS_OFF', 'SCHEDULED_TIME', 'ELAPSED_TIME', 'AIR_TIME', 'DISTANCE',
#       'WHEELS_ON', 'TAXI_IN', 'SCHEDULED_ARRIVAL', 'ARRIVAL_TIME', 'SECURITY_DELAY'], axis = 1) 


flights_model = flights.drop(['MONTH', 'DAY', 'DAY_OF_WEEK', 'AIRLINE', 'FLIGHT_NUMBER',
       'TAIL_NUMBER', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT',
       'SCHEDULED_DEPARTURE', 'DEPARTURE_TIME', 'TAXI_OUT','DISTANCE'], axis = 1) 

 
#####################Calculate z scores to detect outliers##################
for col in flights_model.columns:
    col_zscore = col + '_zscore'
    flights_model[col_zscore] = (flights_model[col] - flights_model[col].mean())/flights_model[col].std(ddof=0)

flights_model.columns   
flights_model['outlier_DEPARTURE_DELAY'] = (abs(flights_model['DEPARTURE_DELAY_zscore'])>3).astype(int)
flights_model['outlier_AIR_SYSTEM_DELAY'] = (abs(flights_model['AIR_SYSTEM_DELAY_zscore'])>3).astype(int)
flights_model['outlier_LATE_AIRCRAFT_DELAY'] = (abs(flights_model['LATE_AIRCRAFT_DELAY_zscore'])>3).astype(int)
flights_model['outlier_WEATHER_DELAY'] = (abs(flights_model['WEATHER_DELAY_zscore'])>3).astype(int)
flights_model['outlier_AIRLINE_DELAY'] = (abs(flights_model['AIRLINE_DELAY_zscore'])>3).astype(int)
flights_model['outlier_ARRIVAL_DELAY'] = (abs(flights_model['ARRIVAL_DELAY_zscore'])>3).astype(int)
flights_model['outlier_WHEELS_OFF'] = (abs(flights_model['WHEELS_OFF_zscore'])>3).astype(int)
flights_model['outlier_SCHEDULED_TIME'] = (abs(flights_model['SCHEDULED_TIME_zscore'])>3).astype(int)
flights_model['outlier_ELAPSED_TIME'] = (abs(flights_model['ELAPSED_TIME_zscore'])>3).astype(int)
flights_model['outlier_AIR_TIME'] = (abs(flights_model['AIR_TIME_zscore'])>3).astype(int)
flights_model['outlier_WHEELS_ON'] = (abs(flights_model['WHEELS_ON_zscore'])>3).astype(int)
flights_model['outlier_TAXI_IN'] = (abs(flights_model['TAXI_IN_zscore'])>3).astype(int)
flights_model['outlier_SCHEDULED_ARRIVAL'] = (abs(flights_model['SCHEDULED_ARRIVAL_zscore'])>3).astype(int)
flights_model['outlier_ARRIVAL_TIME'] = (abs(flights_model['ARRIVAL_TIME_zscore'])>3).astype(int)
flights_model['outlier_SECURITY_DELAY'] = (abs(flights_model['SECURITY_DELAY_zscore'])>3).astype(int)


print("outlier_DEPARTURE_DELAY = " +str(flights_model.outlier_DEPARTURE_DELAY.value_counts()))
print("outlier_AIR_SYSTEM_DELAY = " +str(flights_model.outlier_AIR_SYSTEM_DELAY.value_counts()))
print("outlier_LATE_AIRCRAFT_DELAY = " +str(flights_model.outlier_LATE_AIRCRAFT_DELAY.value_counts()))
print("outlier_LATE_AIRCRAFT_DELAY = " +str(flights_model.outlier_WEATHER_DELAY.value_counts()))   
print("outlier_AIRLINE_DELAY = " +str(flights_model.outlier_AIRLINE_DELAY.value_counts()))   
print("outlier_ARRIVAL_DELAY = " +str(flights_model.outlier_ARRIVAL_DELAY.value_counts()))   

print("outlier_WHEELS_OFF = " +str(flights_model.outlier_WHEELS_OFF.value_counts()))
print("outlier_SCHEDULED_TIME = " +str(flights_model.outlier_SCHEDULED_TIME.value_counts()))
print("outlier_ELAPSED_TIME = " +str(flights_model.outlier_ELAPSED_TIME.value_counts()))
print("outlier_AIR_TIME = " +str(flights_model.outlier_AIR_TIME.value_counts()))   
print("outlier_WHEELS_ON = " +str(flights_model.outlier_WHEELS_ON.value_counts()))   
print("outlier_TAXI_IN = " +str(flights_model.outlier_TAXI_IN.value_counts()))
print("outlier_SCHEDULED_ARRIVAL = " +str(flights_model.outlier_SCHEDULED_ARRIVAL.value_counts()))
print("outlier_ARRIVAL_TIME = " +str(flights_model.outlier_ARRIVAL_TIME.value_counts()))
print("outlier_SECURITY_DELAY = " +str(flights_model.outlier_SECURITY_DELAY.value_counts()))

flights_model.drop(flights_model[flights_model['outlier_DEPARTURE_DELAY'] == 1].index, inplace = True)
flights_model.drop(flights_model[flights_model['outlier_AIR_SYSTEM_DELAY'] == 1].index, inplace = True)
flights_model.drop(flights_model[flights_model['outlier_LATE_AIRCRAFT_DELAY'] == 1].index, inplace = True)
flights_model.drop(flights_model[flights_model['outlier_LATE_AIRCRAFT_DELAY'] == 1].index, inplace = True)
flights_model.drop(flights_model[flights_model['outlier_AIRLINE_DELAY'] == 1].index, inplace = True)
flights_model.drop(flights_model[flights_model['outlier_ARRIVAL_DELAY'] == 1].index, inplace = True)
flights_model.drop(flights_model[flights_model['outlier_WHEELS_OFF'] == 1].index, inplace = True)
flights_model.drop(flights_model[flights_model['outlier_SCHEDULED_TIME'] == 1].index, inplace = True)
flights_model.drop(flights_model[flights_model['outlier_ELAPSED_TIME'] == 1].index, inplace = True)
flights_model.drop(flights_model[flights_model['outlier_AIR_TIME'] == 1].index, inplace = True)
flights_model.drop(flights_model[flights_model['outlier_WHEELS_ON'] == 1].index, inplace = True)
flights_model.drop(flights_model[flights_model['outlier_TAXI_IN'] == 1].index, inplace = True)
flights_model.drop(flights_model[flights_model['outlier_SCHEDULED_ARRIVAL'] == 1].index, inplace = True)
flights_model.drop(flights_model[flights_model['outlier_ARRIVAL_TIME'] == 1].index, inplace = True)
flights_model.drop(flights_model[flights_model['outlier_SECURITY_DELAY'] == 1].index, inplace = True)
###################################################################
flights_model.columns
flights_final = flights_model.drop(['DEPARTURE_DELAY_zscore', 'AIR_SYSTEM_DELAY_zscore',
       'LATE_AIRCRAFT_DELAY_zscore', 'WEATHER_DELAY_zscore', 'AIRLINE_DELAY_zscore', 'ARRIVAL_DELAY_zscore',
       'WHEELS_OFF_zscore','SCHEDULED_TIME_zscore','ELAPSED_TIME_zscore','AIR_TIME_zscore','WHEELS_ON_zscore',
       'TAXI_IN_zscore','SCHEDULED_ARRIVAL_zscore','ARRIVAL_TIME_zscore','SECURITY_DELAY_zscore',
       'outlier_DEPARTURE_DELAY', 'outlier_AIR_SYSTEM_DELAY',
       'outlier_LATE_AIRCRAFT_DELAY', 'outlier_WEATHER_DELAY', 'outlier_AIRLINE_DELAY', 'outlier_ARRIVAL_DELAY',
       'outlier_WHEELS_OFF','outlier_SCHEDULED_TIME','outlier_ELAPSED_TIME','outlier_AIR_TIME','outlier_WHEELS_ON',
       'outlier_TAXI_IN','outlier_SCHEDULED_ARRIVAL','outlier_ARRIVAL_TIME','outlier_SECURITY_DELAY'], axis = 1) 
flights_final = flights_final.drop(['AIR_SYSTEM_DELAY','WHEELS_ON','SCHEDULED_ARRIVAL','ARRIVAL_TIME','SECURITY_DELAY'],axis = 1)
flights_final.columns
flights_final.shape
flights_final.isin([0]).sum()  
a =flights_final.tail(5)

###############################################################################
### Splitting the data into train and test data 

from sklearn.model_selection import train_test_split
from sklearn import linear_model

train,test  = train_test_split(flights_final,test_size = 0.2) # 20% size
trainX = train.drop(['DEPARTURE_DELAY'], axis = 1)
trainY = train['DEPARTURE_DELAY']
# preparing the model on train data 

X2 = sm.add_constant(trainX)
est = sm.OLS(trainY, X2)
est2 = est.fit()
print(est2.summary())


regr = linear_model.LinearRegression()
regr.fit(trainX,trainY)

pickle.dump(regr,open('modelLR.pkl','wb'))

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

predictions = regr.predict(trainX) 
# Observed values VS Fitted values
plt.scatter(trainY,predictions,c="r");plt.xlabel("observed_values");plt.ylabel("fitted_values")

testX = test.drop(['DEPARTURE_DELAY'], axis = 1)
testY = test['DEPARTURE_DELAY']
predictions = regr.predict(testX) 

# Observed values VS Fitted values
plt.scatter(testY,predictions,c="r");plt.xlabel("observed_values");plt.ylabel("fitted_values")


# test residual values 
test_resid  = predictions - testY

# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid*test_resid))
# Checking Residuals are normally distributed
st.probplot(test_resid, dist="norm", plot=pylab)

# Residuals VS Fitted Values 
plt.scatter(predictions,test_resid,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")
score = metrics.mean_squared_error(predictions, testY)
print("Mean squared error = ", score)

# R Squared
r2 = metrics.r2_score(testY.values.ravel(), predictions)

# train residual values 
train_resid  = predictions - trainY

# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid*train_resid))
score = metrics.mean_squared_error(predictions, trainY)
print("Mean squared error = ", score)

# R Squared
r2 = metrics.r2_score(trainY.values.ravel(), predictions)
#############################################################
###POLYNOMIAL
#############################################################

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# PolynomialFeatures (prepreprocessing)
poly = PolynomialFeatures(degree=3)
trainPoly = poly.fit_transform(trainX)
testPoly = poly.fit_transform(testX)
trainX.columns

# Instantiate
lg = LinearRegression()

# Fit
lg.fit(trainPoly, trainY)
pred = lg.predict(trainPoly)
# Obtain coefficients
lg.coef_
pickle.dump(lg,open('modelPoly.pkl','wb'))
# Predict
pred = lg.predict(testPoly)



#------------------------Gridsearch CV-----------------------------
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
#from sklearn.model_selection import cross_val_score 
degrees = [2, 3, 4, 5, 6] # Change degree "hyperparameter" here
normalizes = [True, False] # Change normalize hyperparameter here
best_score = 0
best_degree = 0
for degree in degrees:
    for normalize in normalizes:
        poly_features = PolynomialFeatures(degree = degree)
        X_train_poly = poly_features.fit_transform(trainX)
        polynomial_regressor = LinearRegression(normalize=normalize)
        polynomial_regressor.fit(X_train_poly, trainY)
        scores = cross_val_score(polynomial_regressor, X_train_poly, trainY, cv=3) # Change k-fold cv value here
        if max(scores) > best_score:
            best_score = max(scores)
            best_degree = degree
            best_normalize = normalize
print(best_score)
print(best_normalize)
print(best_degree)

poly_features = PolynomialFeatures(degree = best_degree)
X_train_poly = poly_features.fit_transform(trainX)
best_polynomial_regressor = LinearRegression(normalize=best_normalize)
pred = polynomial_regressor.fit(X_train_poly, trainY)
pred = polynomial_regressor.predict(X_train_poly)
pickle.dump(poly_features,open('modelPoly.pkl','wb'))

X_test_poly = poly_features.fit_transform(testX)
pred = polynomial_regressor.predict(X_test_poly)
pred = polynomial_regressor.predict(X_test_poly)

# train residual values 
train_resid  = pred  - trainY

# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid*train_resid))
score = metrics.mean_squared_error(pred, trainY)
print("Mean squared error = ", score)
# R Squared
r2 = metrics.r2_score(trainY.values.ravel(), pred)
#-------------------------------------------------------------------

# Observed values VS Fitted values
plt.scatter(testY,pred,c="r");plt.xlabel("observed_values");plt.ylabel("fitted_values")


# test residual values 
test_resid  = pred - testY

# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid*test_resid))

# R Squared
r2 = metrics.r2_score(testY.values.ravel(), pred)

# Checking Residuals are normally distributed
st.probplot(test_resid, dist="norm", plot=pylab)

# Residuals VS Fitted Values 
plt.scatter(pred,test_resid,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")




score = metrics.mean_squared_error(pred, testY)
print("Mean squared error = ", score)
###################################################################################


############################################################################
##############Ridge Regression
################################################################################

from sklearn.linear_model import Ridge
ridgereg = Ridge(alpha=0 ,normalize=True)
poly = PolynomialFeatures(degree = 2)
X_ = poly.fit_transform(trainX)
ridgereg.fit(X_, trainY)
result = ridgereg.predict(X_)


pickle.dump(ridgereg,open('modelRidge.pkl','wb'))
 
X_ = poly.fit_transform(testX)
result = ridgereg.predict(X_)
score = metrics.mean_squared_error(result, testY)
print("Mean squared error = ", score)

# test residual values 
test_resid  = result - testY

# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid*test_resid))

# R Squared
r2 = metrics.r2_score(testY.values.ravel(), result)


# Checking Residuals are normally distributed
st.probplot(test_resid, dist="norm", plot=pylab)

# Residuals VS Fitted Values 
plt.scatter(result,test_resid,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")

# Observed values VS Fitted values
plt.scatter(testY,result,c="r");plt.xlabel("observed_values");plt.ylabel("fitted_values")



score_min = 10000
for pol_order in range(1, 3):
    for alpha in range(0, 20, 2):
        ridgereg = Ridge(alpha = alpha/10, normalize=True)
        poly = PolynomialFeatures(degree = pol_order)
        regr = linear_model.LinearRegression()
        X_ = poly.fit_transform(trainX)
        ridgereg.fit(X_, trainY)        
        X_ = poly.fit_transform(testX)
        result = ridgereg.predict(X_)
        score = metrics.mean_squared_error(result, testY)        
        if score < score_min:
            score_min = score
            parameters = [alpha/10, pol_order]
        print("n={} alpha={} , MSE = {:<0.5}".format(pol_order, alpha, score))
 
#result = pd.DataFrame(result)
# train residual values 
train_resid  = result  - trainY
result.shape
trainY.shape
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid*train_resid))
score = metrics.mean_squared_error(result, trainY)
print("Mean squared error = ", score)
# R Squared
r2 = metrics.r2_score(trainY.values.ravel(), result)       
################################################################################
        
################################################################################
##############Random Forest Regression############################
################################################################################

from sklearn.ensemble import RandomForestRegressor
#from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_predict
# define dataset
# Perform Grid-Search
gsc = GridSearchCV(
        estimator=RandomForestRegressor(),
        param_grid={
            'max_depth': range(3,7),
            'n_estimators': (10, 50, 100, 1000),
        },
        cv=15, scoring='neg_mean_squared_error', verbose=0,n_jobs=-1)
    
grid_result = gsc.fit(trainX, trainY)
best_params = grid_result.best_params_
    
rfr = RandomForestRegressor(max_depth=best_params["max_depth"], 
                n_estimators=best_params["n_estimators"], random_state=False, verbose=False)# Perform K-Fold CV
rfr.fit(trainX, trainY)  
pickle.dump(rfr,open('modelRandomF.pkl','wb'))

scores = cross_val_score(rfr, trainX, trainY, cv=5, scoring='neg_mean_absolute_error')
scores.mean()
predictions = rfr.predict(trainX)
predictions = cross_val_predict(rfr, trainX, trainY, cv=5)
predictions = cross_val_predict(rfr, testX, testY, cv=5)
predictions = rfr.predict(testX)
score = metrics.mean_squared_error(predictions, testY)
print("Mean squared error = ", score)


# test residual values 
test_resid  = predictions - testY

# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid*test_resid))

# R Squared
r2 = metrics.r2_score(testY.values.ravel(), predictions)


# Checking Residuals are normally distributed
st.probplot(test_resid, dist="norm", plot=pylab)

# Residuals VS Fitted Values 
plt.scatter(predictions,test_resid,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")

# Observed values VS Fitted values
plt.scatter(testY,predictions,c="r");plt.xlabel("observed_values");plt.ylabel("fitted_values")

f1score = metrics.f1_score(predictions,trainY)



# train residual values 
train_resid  = predictions  - trainY

# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid*train_resid))
score = metrics.mean_squared_error(predictions, trainY)
print("Mean squared error = ", score)
# R Squared
r2 = metrics.r2_score(trainY.values.ravel(), predictions)  
####################################################################


################Tenserflow Model Linear Regression####################
import tensorflow as tf

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1,input_shape=[4]))
model.compile(loss='mean_squared_error',optimizer=tf.keras.optimizers.Adam(0.01))

model.summary()

history = model.fit(trainX,trainY,epochs = 100)

plt.plot(history.history['loss'])
plt.show()

predictions = model.predict(testX)
predictions  = pd.DataFrame(predictions)
 
# test residual values 
test_resid  = predictions - testY

# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid*test_resid))

# R Squared
r2 = metrics.r2_score(testY.values.ravel(), predictions)


# Checking Residuals are normally distributed
st.probplot(test_resid, dist="norm", plot=pylab)

# Residuals VS Fitted Values 
plt.scatter(predictions,test_resid,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")

# Observed values VS Fitted values
plt.scatter(testY,predictions,c="r");plt.xlabel("observed_values");plt.ylabel("fitted_values")

#######################################################3

# Instantiate
abc = AdaBoostClassifier()

# Fit
abc.fit(trainX, trainY)

# Predict
prediction = abc.predict(trainX)

prediction = abc.predict(testX)

# train residual values 
train_resid  = prediction  - trainY

# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid*train_resid))
score = metrics.mean_squared_error(prediction, trainY)
print("Mean squared error = ", score)
# R Squared
r2 = metrics.r2_score(trainY.values.ravel(), prediction) 


test_resid  = prediction  - testY
# RMSE value for train data 
test_rmse = np.sqrt(np.mean(test_resid*test_resid))
score = metrics.mean_squared_error(prediction, testY)
print("Mean squared error = ", score)
# R Squared
r2 = metrics.r2_score(testY.values.ravel(), prediction) 

###################################################################


# Instantiate
abc = AdaBoostClassifier(DecisionTreeClassifier(criterion = 'entropy'),n_estimators = 1000)
 
# Fit
abc.fit(trainX, trainY)

# Predict
prediction = abc.predict(trainX)

prediction = abc.predict(testX)

# train residual values 
train_resid  = prediction  - trainY

# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid*train_resid))
score = metrics.mean_squared_error(prediction, trainY)
print("Mean squared error = ", score)
# R Squared
r2 = metrics.r2_score(trainY.values.ravel(), prediction) 


test_resid  = prediction  - testY
# RMSE value for train data 
test_rmse = np.sqrt(np.mean(test_resid*test_resid))
score = metrics.mean_squared_error(prediction, testY)
print("Mean squared error = ", score)
# R Squared
r2 = metrics.r2_score(testY.values.ravel(), prediction) 




#######################################################################