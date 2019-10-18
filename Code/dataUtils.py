import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

from os import getpid
from subprocess import Popen
from multiprocessing import current_process

import warnings
warnings.filterwarnings("ignore")

class Data:

    def __init__(self, scaledTimeSeries, scaler):

        self.scaledTimeSeries = scaledTimeSeries
        self.scaler = scaler

        self.noTimeSeries = self.scaledTimeSeries.shape[1]
        self.maxLookBack = 25
        self.splitIndex = self.scaledTimeSeries.shape[0] - self.maxLookBack

    def createLSTMDataSet(self, lookBack):

        # Ensure that all models start at the same datapoint
        if self.maxLookBack < lookBack: 
            self.maxLookBack = lookBack
            self.createHARDataSet()

        dataX, dataY = np.zeros([1, lookBack, self.noTimeSeries]), \
                        np.zeros([1, self.noTimeSeries])

        for i in range(self.maxLookBack, len(self.scaledTimeSeries) - 1):

            xlookback = np.expand_dims(self.scaledTimeSeries[i - lookBack:i], axis = 0)
            dataX = np.concatenate((dataX, xlookback), axis = 0)

            yForecast = self.scaledTimeSeries[i:i+1]
            dataY = np.concatenate((dataY, yForecast), axis = 0)

            assert np.sum(np.subtract(np.concatenate((xlookback[-1],yForecast),axis = 0), \
                    self.scaledTimeSeries[i - lookBack:i+1])) == 0

        self.xLSTM = dataX.reshape((dataX.shape[0], dataX.shape[1],self.noTimeSeries))[1:]
        self.yLSTM = dataY.reshape((dataY.shape[0], dataY.shape[1]))[1:]
        
        return

    def createHARDataSet(self, timeSeriesInput = None):

        if timeSeriesInput is not None: 
            timeSeriesData = timeSeriesInput
        else: timeSeriesData = self.scaledTimeSeries
            
        yVector, xDay, xWeek, xMonth = np.zeros((1, self.noTimeSeries)),\
                                                np.zeros((1, self.noTimeSeries)),\
                                                np.zeros((1, self.noTimeSeries)),\
                                                np.zeros((1, self.noTimeSeries))

        for i in range(self.maxLookBack, timeSeriesData.shape[0] - 1):
            
            yVector = np.append(yVector, timeSeriesData[[i+1],:], axis = 0)

            xDay = np.append(xDay, timeSeriesData[[i],:], axis = 0) 
            xWeek = np.append(xWeek, np.mean(timeSeriesData[i-6:i,:], 
                                axis = 0, keepdims= True), axis = 0)
            xMonth = np.append(xMonth, np.mean(timeSeriesData[i-23:i,:], 
                                axis = 0, keepdims = True), axis = 0)

        dataHAR = { 'y':        yVector[1:], 
                    'xDay':     xDay[1:], 
                    'xWeek':    xWeek[1:], 
                    'xMonth':   xMonth[1:]}
        
        if timeSeriesInput is None: self.dataHAR = dataHAR
        else: return dataHAR

        
    def splitData(self, splitIndex, startPointIndex = 0):
        self.splitIndex = splitIndex
        self.startPointIndex = startPointIndex

    def xTrain(self):
        return self.scaledTimeSeries[self.startPointIndex + 
                self.maxLookBack:self.splitIndex - 1]
    def yTrain(self):
        return self.scaledTimeSeries[self.startPointIndex + 
                self.maxLookBack + 1: self.splitIndex]
    def xTest(self):
        return self.scaledTimeSeries[self.splitIndex - 1:-1]
    def yTest(self):
        return self.scaledTimeSeries[self.splitIndex + 1:]

    def xTrainLSTM(self):
        return self.xLSTM[self.startPointIndex:self.splitIndex]
    def yTrainLSTM(self):
        return self.yLSTM[self.startPointIndex:self.splitIndex]
    def xTestLSTM(self):
        return self.xLSTM[self.splitIndex:]
    def yTestLSTM(self):
        return self.yLSTM[self.splitIndex:]

    def dataHARtrain(self):
        dataHARtrain = dict.fromkeys(self.dataHAR)
        for dataSeries in self.dataHAR:
            dataHARtrain[dataSeries] = \
                self.dataHAR[dataSeries][self.startPointIndex:self.splitIndex]
        return dataHARtrain
    
    def dataHARtest(self):
        dataHARtest= dict.fromkeys(self.dataHAR)
        for dataSeries in self.dataHAR:
            dataHARtest[dataSeries] = \
                self.dataHAR[dataSeries][self.splitIndex:]
        return dataHARtest


class ErrorMetrics:

    def __init__(self, errorVectors, errorMatrices):

        self.errorVector = errorVectors
        self.errorMatrix = errorMatrices

def dataPrecprocessingUnivariate(path, fileName, rawData = None):

    if rawData == None:
        rawData = pd.read_csv(path + fileName, 
                                skiprows = 1, 
                                sep = "\,", 
                                engine ='python')
    rawData.columns = rawData.columns.str.replace('"','')
    rawData.head()
    rawData['Dates'] = pd.to_datetime(rawData['Dates'], format = '%Y%m%d').dt.date
    rawData = rawData.set_index('Dates')

    preprocessedData = rawData[['DJI_rv', 
                                'FTSE_rv', 
                                'GDAXI_rv', 
                                'N225_rv', 
                                'EUR_rv']]

    preprocessedData.columns = preprocessedData.columns.str.replace("_rv", "")

    preprocessedData = preprocessedData.interpolate(limit = 3).dropna()

    for i in range(0, len(preprocessedData.columns)):
        preprocessedData["Log" + preprocessedData.columns[i]] = np.log(preprocessedData[preprocessedData.columns[i]])

    return preprocessedData


def loadScaleData(assetList, path, fileName):
    
    preprocessedData = dataPrecprocessingUnivariate(path, fileName)

    timeSeries = np.zeros(np.array(preprocessedData["Log" + assetList[0]]).reshape(-1, 1).shape)
    for asset in assetList:
        timeSeries = np.hstack((timeSeries, np.array(preprocessedData["Log" + asset]).reshape(-1, 1)))
    timeSeries = timeSeries[:,1:]

    scaler = MinMaxScaler(feature_range = (0.0, 1))
    scaledTimeSeries = scaler.fit_transform(timeSeries)

    return Data(scaledTimeSeries, scaler)
        
def calculateErrorVectors(data, model, 
                                forecastHorizon, 
                                windowMode, 
                                windowSize = None, 
                                silent = True,
                                startInd = 25):

    # When using multiprocessing, limit CPU Usage depending of model Type
    if current_process().name is not "MainProcess":
        if model.modelType == "LSTM": limitCPU(200)
        elif model.modelType == "ESN": limitCPU(200)
        elif model.modelType == "HAR": limitCPU(20)

    finalInd = data.scaledTimeSeries.shape[0] - forecastHorizon - data.maxLookBack
    finalInd = startInd + 250
    assert windowMode.upper() in ["EXPANDING", "ROLLING", "FIXED"], "Window Mode not recognized"
    
    errorMatrices = {"RMSE": [],
                        "QLIK": [],
                        "L1Norm": []}

    avgErrorVectors = {"RMSE": None,
                        "QLIK": None,
                        "L1Norm": None}

    def modelForecast(model, index):

        forecast, actual = model.multiStepAheadForecast(data, 
                                                        forecastHorizon, 
                                                        index, 
                                                        windowMode, 
                                                        windowSize)

        if data.scaler is not None:
            forecast = data.scaler.inverse_transform(forecast)
            actual = data.scaler.inverse_transform(actual)

        assert actual.shape == forecast.shape, "actual " + str(actual.shape) +  \
            " and forecast shape " + str(forecast.shape) + " do not match"
        
        return actual, forecast
    
    for index in range(startInd, finalInd):

        if silent is False and (index % 25 == 0 or index == finalInd - 1): 
            print(model.modelType + " Evaluation is at index: " + str(index) + "/ " \
            + str(finalInd - 1))
        
        actual, forecast = modelForecast(model, index)
        
        def calculateForecastingError(errorType):

            if errorType == "RMSE":
                errorVector = np.mean(np.power(np.exp(forecast) - np.exp(actual),2), 
                                axis = 1, keepdims = True)
            elif errorType == "QLIK":
                errorVector = np.mean(np.log(np.exp(actual) / np.exp(forecast)) + \
                                np.exp(actual) / np.exp(forecast), 
                                axis = 1, keepdims = True)
            elif errorType == "L1Norm":
                errorVector = np.linalg.norm((forecast - actual), axis = 1, keepdims = True)
            
            errorMatrices[errorType].append(errorVector)

            if errorType == "RMSE":
                avgErrorVectors[errorType] = np.sqrt(np.average(errorMatrices[errorType], axis = 0))
            elif errorType in ["QLIK", "L1Norm"]:
                avgErrorVectors[errorType] = np.average(errorMatrices[errorType], axis = 0)
                
        calculateForecastingError("RMSE")
        calculateForecastingError("QLIK")
        calculateForecastingError("L1Norm")

        # Debug Function
        def showForecast(errorType):
            
            avgErrorVector = avgErrorVectors[errorType]
            errorMatrix = errorMatrices[errorType]
           
            ax1 = plt.subplot(3, 1, 1)
            ax1.set_title("Actual vs Forecast")
            ax1.plot(np.exp(forecast), label = "Forecast")
            ax1.plot(np.exp(actual), label = "Actual")
            ax1.legend()

            ax2 = plt.subplot(3, 1, 2)
            for i in range(0,len(errorMatrix)):
                shade = str(i/(len(errorMatrix)+0.1))
                ax2.plot(np.sqrt(errorMatrix[i]), color=shade, linestyle='dotted')
            ax2.plot(avgErrorVector, color='blue', marker ="x" )
            ax2.set_title("Error Vectors.")

            ax3 = plt.subplot(3, 1, 3)
            ax3.set_title("Error Vector Avg. Index: " +  str(index))
            ax3.plot(avgErrorVector, color='blue', marker ="x" )

            plt.tight_layout()
            plt.show()

    return ErrorMetrics(avgErrorVectors, errorMatrices)

def limitCPU(cpuLimit):

    try:
        limitCommand = "cpulimit --pid " + str(getpid()) + " --limit " + str(cpuLimit)
        Popen(limitCommand, shell=True)
        print("CPU Limit at " + str(cpuLimit))
    except:
        print("Limiting CPU Usage failed")



