import statsmodels.api as sm
import numpy as np
from pandas import DataFrame
import dataUtils

from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings("ignore")

class HARmodel:

    def __init__(self):

        self.fitted = False
        self.modelType = "HAR"

    def fit(self, dataHAR, silent = True):

        Y = dataHAR['y']
        X = sm.add_constant(np.concatenate([dataHAR['xDay'], 
                                            dataHAR['xWeek'], 
                                            dataHAR['xMonth']], 
                                            axis = 1))

        modelSetUp = sm.OLS(Y, X)
        self.model = modelSetUp.fit()

        if not silent:
            print(self.model.summary())

        self.fitted = True

    def multiStepAheadForecast(self, data, forecastHorizon, index, 
                                            windowMode, windowSize = 0):
        
        if windowMode.upper() == "EXPANDING":
            data.splitData(index, startPointIndex = 0)
            self.fit(data.dataHARtrain())
        elif windowMode.upper() == "ROLLING":
            data.splitData(index, startPointIndex = index - windowSize)
            self.fit(data)
        elif windowMode.upper() == "FIXED":
            data.splitData(index, startPointIndex = 0)
        
        actual = data.dataHARtest()['y'][:forecastHorizon]
        assert actual.shape == (forecastHorizon,data.noTimeSeries)
    
        def recursiveForecast(forecast, backlog):

            dataHAR = data.createHARDataSet(backlog[-60:])

            X = sm.add_constant(np.concatenate([dataHAR['xDay'], 
                                                dataHAR['xWeek'], 
                                                dataHAR['xMonth']], 
                                                axis = 1))

            oneDayAheadForecast = self.model.predict(X[-1]).reshape(1,-1)

            backlog = np.concatenate([backlog, oneDayAheadForecast])
            forecast = np.concatenate([forecast, oneDayAheadForecast])

            return recursiveForecast(forecast, backlog) if forecast.shape[0] - 1 < forecastHorizon \
                else forecast[1:]

        multiStepForecast = recursiveForecast(np.zeros((1,data.noTimeSeries)), 
                                data.dataHARtrain()['xDay'])
        
        return multiStepForecast, actual