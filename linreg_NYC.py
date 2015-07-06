import numpy as np
import pandas
import statsmodels.api as sm
import statsmodels
from datetime import date
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import stats
from ggplot import *

"""
1) implement the linear_regression() procedure
2) Select features (in the predictions procedure) and make predictions.

"""

def linear_regression(features, values):
    """
    Perform linear regression given a data set with an arbitrary number of features.
    
    This can be the same code as in the lesson #3 exercise.
    """
    
    features = sm.add_constant(features)
    
    model = sm.OLS(values, features)
    results = model.fit()
    intercept = results.params[0]
    params = results.params[1:]
    #print results.params
    #print intercept
    
    return intercept, params

def predictions(dataframe):
    '''
    The NYC turnstile data is stored in a pandas dataframe called weather_turnstile.
    Using the information stored in the dataframe, let's predict the ridership of
    the NYC subway using linear regression with gradient descent.
    
    You can download the complete turnstile weather dataframe here:
    https://www.dropbox.com/s/meyki2wl9xfa7yk/turnstile_data_master_with_weather.csv    
    

    """for i in range(dataframe.shape[0]):
        try:
            dataframe.loc[i,'weekday'] = date.weekday(datetime.strptime(dataframe.loc[i,'DATEn'],'%m/%d/%Y'))
        except:
            try:
                dataframe.loc[i,'weekday'] = date.weekday(datetime.strptime(dataframe.loc[i,'DATEn'],'%Y-%m-%d'))
            except:
                None
                #print '%ith conversion failed' % i"""
    features = dataframe[['rain', 'precipi', 'hour', 'meantempi', 'wspdi', 'meanpressurei']]
    dummy_units = pandas.get_dummies(dataframe['UNIT'], prefix='unit')
    features = features.join(dummy_units)
    dummy_units = pandas.get_dummies(dataframe['station'], prefix='station')
    features = features.join(dummy_units)
    dummy_units = pandas.get_dummies(dataframe['conds'], prefix='condition')
    features = features.join(dummy_units)
    dummy_units = pandas.get_dummies(dataframe['day_week'], prefix='day_week')
    features = features.join(dummy_units)
    
    #features.to_csv(r"C:\Users\nb66827\Desktop\p1features.csv", index=False)
	
    # Values
    values = dataframe['ENTRIESn_hourly']

    # Perform linear regression
    intercept, params = linear_regression(features, values)
    
    predictions = intercept + np.dot(features, params)
    return predictions

def compute_r_squared(data, predictions):
    # Write a function that, given two input numpy arrays, 'data', and 'predictions,'
    # returns the coefficient of determination, R^2, for the model that produced 
    # predictions.
    # 
    # Numpy has a couple of functions -- np.mean() and np.sum() --
    # that you might find useful, but you don't have to use them.

    # YOUR CODE GOES HERE
    #print data, predictions
    num = data-predictions
    data_mean = np.mean(data)
    den = data-data_mean
    r_squared = 1 - (num**2).sum()/(den**2).sum()

    return r_squared

def run_p1():
	df = pandas.read_csv(r'C:\Users\nb66827\Downloads\improved-dataset\turnstile_weather_v2.csv')
	preds = predictions(df)
	r2 = compute_r_squared(df['ENTRIESn_hourly'], preds)
	#print r2
	resdf = df['ENTRIESn_hourly'] - preds
	#plt.plot(resdf)
	#plt.show()
	return resdf




