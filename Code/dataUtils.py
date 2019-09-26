import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

from os import getpid
from subprocess import Popen
from multiprocessing import current_process

class Data:

    def __init__(self, xTrain, yTrain, xTest, yTest, scaler):

        self.xTrain = xTrain
        self.yTrain = yTrain
        self.xTest = xTest
        self.yTest = yTest
        self.scaler = scaler
    
    def __convertDataLSTM(self, xVector, lookBack, forecastHorizon):

        dataX, dataY = np.empty([1,lookBack]), np.empty([1,forecastHorizon])

        for i in range(lookBack, len(xVector) - 1 - forecastHorizon):

            x_lookback = xVector[i - lookBack:i].T
            dataX = np.concatenate((dataX, x_lookback), axis = 0)

            y_multistepForecast = xVector[i: i + forecastHorizon].T
            dataY = np.concatenate((dataY, y_multistepForecast), axis = 0)

            assert np.sum(np.subtract(np.concatenate((x_lookback.T,y_multistepForecast.T)), \
                    xVector[i - lookBack:i + forecastHorizon])) == 0

        dataX = dataX.reshape((dataX.shape[0], dataX.shape[1],1))[1:]
        dataY = dataY.reshape((dataY.shape[0], dataY.shape[1]))[1:]

        return dataX, dataY

    def createLSTMDataSet(self,lookBack, forecastHorizon):

        self.xTrainLSTM, self.yTrainLSTM = self.__convertDataLSTM(self.xTrain, lookBack, forecastHorizon)
        self.xTestLSTM, self.yTestLSTM = self.__convertDataLSTM(self.xTest, lookBack, forecastHorizon)


class ErrorMetrics:

    def __init__(self, errorVectors, errorMatrices):

        self.errorVector = errorVectors
        self.errorMatrix = errorMatrices

def dataPrecprocessingUnivariate(path, fileName, rawData = None):

    if rawData == None:
        rawData = pd.read_csv(path + fileName, skiprows = 1, sep="\,", engine='python')
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


def loadScaleDataUnivariate(asset, path, fileName, scaleData = True, rawData = None):

    scaler = None

    preprocessedData = dataPrecprocessingUnivariate(path, fileName)

    timeSeries = np.array(preprocessedData["Log" + asset]).reshape(len(preprocessedData["Log" + asset]), 1)

    nTrainingExamples = int((timeSeries.shape[0])*0.8)
    yTrain = timeSeries[1:nTrainingExamples].reshape(nTrainingExamples - 1,1)
    xTrain = timeSeries[:nTrainingExamples-1].reshape(nTrainingExamples -1,1)

    yTest = timeSeries[nTrainingExamples + 1:].reshape(timeSeries.shape[0] - nTrainingExamples -1,1)
    xTest = timeSeries[nTrainingExamples:-1].reshape(timeSeries.shape[0] - nTrainingExamples -1,1)

    if scaleData: 
        scaler = MinMaxScaler(feature_range = (0.0, 1))
        xTrain = scaler.fit_transform(xTrain.reshape(-1, 1))
        yTrain = scaler.transform(yTrain.reshape(-1, 1))

        yTest = scaler.transform(yTest.reshape(-1, 1))
        xTest = scaler.transform(xTest.reshape(-1, 1))

    assert xTrain.shape == yTrain.shape
    assert xTest.shape == yTest.shape

    return Data(xTrain, yTrain, xTest, yTest, scaler)

         
def convertHAR(xVector):

    yVector, xVector1, xVector5, xVector22 = np.array([]), np.array([]), np.array([]), np.array([]) 

    for i in range(22, len(xVector) - 1):
        
        yVector = np.append(yVector, xVector[i+1])

        xVector1 = np.append(xVector1, xVector[i]) 
        xVector5 = np.append(xVector5, np.mean(xVector[i - 5:i]))
        xVector22 = np.append(xVector22, np.mean(xVector[i - 22:i]))

    dataHAR = pd.DataFrame(data = {'Y': yVector, 'X1': xVector1, 'X5': xVector5, 'X22': xVector22})
    dataHAR = dataHAR.dropna()

    return dataHAR

def calculateErrorVectors(data, model, forecastHorizon, windowMode, windowSize = None, silent = True):
    
    # When using multiprocessing, limit CPU Usage depending of model Type
    if current_process().name is not "MainProcess":
        if model.modelType == "LSTM": limitCPU(200)
        elif model.modelType == "ESN": limitCPU(200)
        elif model.modelType == "HAR": limitCPU(20)

    totalIterations = data.xTest.shape[0] - forecastHorizon + 1
    assert totalIterations > 25, "Increase Test Size of the Test Set"
    
    errorMatrices = {"RMSE": [],
                        "QLIK": [],
                        "L1Norm": []}

    avgErrorVectors = {"RMSE": None,
                        "QLIK": None,
                        "L1Norm": None}

    # Get the forecast and the actual values 
    def modelForecast(model, startIndex):

        forecast = model.multiStepAheadForecast(data, 
                                                forecastHorizon, 
                                                startIndex, 
                                                windowMode, 
                                                windowSize)

        if model.modelType == "HAR" or model.modelType == "ESN":
            actual = data.yTest[startIndex : startIndex + forecastHorizon]

        if model.modelType == "LSTM":
            if data.yTest[startIndex:startIndex + 1].shape == (1,1):
                actual = data.yTest[startIndex:startIndex + forecastHorizon].reshape(1, -1).T
            else: 
                actual = data.yTest[startIndex:startIndex + 1].reshape(1, -1).T

        if data.scaler is not None:
            forecast = data.scaler.inverse_transform(forecast)
            actual = data.scaler.inverse_transform(actual)

        assert actual.shape == forecast.shape, "actual " + str(actual.shape) +  \
            " and forecast shape " + str(forecast.shape) + " do not match"
        
        return actual, forecast

    # Iterate through the Test set and collect the errors
    for startIndex in range(25, totalIterations):

        if silent is False and (startIndex % 25 == 0 or startIndex == totalIterations - 1): 
            print(model.modelType + " Evaluation is at index: " + str(startIndex) + "/ " \
            + str(totalIterations - 1))
        
        actual, forecast = modelForecast(model, startIndex)
        
        def calculateForecastingError(errorType):

            if errorType == "RMSE":
                errorVector = np.power(np.exp(forecast) - np.exp(actual),2)
            elif errorType == "QLIK":
                errorVector = np.log(np.exp(actual) / np.exp(forecast)) + np.exp(actual) / np.exp(forecast)
            elif errorType == "L1Norm":
                errorVector =np.linalg.norm((forecast - actual), axis = 1)
            
            errorMatrices[errorType].append(errorVector)

            if errorType == "RMSE":
                avgErrorVectors[errorType] = np.sqrt(np.average(errorMatrices[errorType], axis = 0))
            elif errorType in ["QLIK", "L1Norm"]:
                avgErrorVectors[errorType] = np.average(errorMatrices[errorType], axis = 0)
                
        calculateForecastingError("RMSE")
        calculateForecastingError("QLIK")
        calculateForecastingError("L1Norm")

        # Debug Function
        def showForecast(avgErrorVector, errorMatrix):
           
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
            ax3.set_title("Error Vector Avg. Index: " +  str(startIndex))
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





