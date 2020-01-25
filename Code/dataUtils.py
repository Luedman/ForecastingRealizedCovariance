import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from scipy.linalg import expm as matrixExponetial
from scipy.linalg import logm as matrixLogarithm
from scipy.stats import entropy
from matplotlib import pyplot as plt
from itertools import combinations_with_replacement
import datetime as dt

from os import getpid
from subprocess import Popen
from multiprocessing import current_process

import warnings

warnings.filterwarnings("ignore")


class Data:
    def __init__(self, scaledTimeSeries, scaler, noAssets, nameDataSet, index):

        self.scaledTimeSeries = scaledTimeSeries
        self.scaler = scaler
        self.nameDataSet = nameDataSet
        self.noAssets = noAssets
        self.index = index

        self.noTimeSeries = self.scaledTimeSeries.shape[1]
        self.maxLookBack = 25
        self.splitIndex = self.scaledTimeSeries.shape[0] - self.maxLookBack

    def createLSTMDataSet(self, lookBack):

        # Ensure that all models start at the same datapoint
        if self.maxLookBack < lookBack:
            self.maxLookBack = lookBack
            self.createHARDataSet()

        dataX, dataY = (
            np.zeros([1, lookBack, self.noTimeSeries]),
            np.zeros([1, self.noTimeSeries]),
        )

        for i in range(self.maxLookBack, len(self.scaledTimeSeries) - 1):

            xlookback = np.expand_dims(self.scaledTimeSeries[i - lookBack : i], axis=0)
            dataX = np.concatenate((dataX, xlookback), axis=0)

            yForecast = self.scaledTimeSeries[i : i + 1]
            dataY = np.concatenate((dataY, yForecast), axis=0)

            assert (
                np.sum(
                    np.subtract(
                        np.concatenate((xlookback[-1], yForecast), axis=0),
                        self.scaledTimeSeries[i - lookBack : i + 1],
                    )
                )
                == 0
            )

        self.xLSTM = dataX.reshape((dataX.shape[0], dataX.shape[1], self.noTimeSeries))[
            1:
        ]
        self.yLSTM = dataY.reshape((dataY.shape[0], dataY.shape[1]))[1:]

        return

    def createHARDataSet(self, timeSeriesInput=None):

        if timeSeriesInput is not None:
            timeSeriesData = timeSeriesInput
        else:
            timeSeriesData = self.scaledTimeSeries

        yVector, xDay, xWeek, xMonth = (
            np.zeros((1, self.noTimeSeries)),
            np.zeros((1, self.noTimeSeries)),
            np.zeros((1, self.noTimeSeries)),
            np.zeros((1, self.noTimeSeries)),
        )

        for i in range(self.maxLookBack, timeSeriesData.shape[0] - 1):

            yVector = np.append(yVector, timeSeriesData[[i + 1], :], axis=0)

            xDay = np.append(xDay, timeSeriesData[[i], :], axis=0)
            xWeek = np.append(
                xWeek,
                np.mean(timeSeriesData[i - 6 : i, :], axis=0, keepdims=True),
                axis=0,
            )
            xMonth = np.append(
                xMonth,
                np.mean(timeSeriesData[i - 23 : i, :], axis=0, keepdims=True),
                axis=0,
            )

        dataHAR = {
            "y": yVector[1:],
            "xDay": xDay[1:],
            "xWeek": xWeek[1:],
            "xMonth": xMonth[1:],
        }

        if timeSeriesInput is None:
            self.dataHAR = dataHAR
        else:
            return dataHAR

    def splitData(self, splitIndex, startPointIndex=0):
        self.splitIndex = splitIndex
        self.startPointIndex = startPointIndex
        testingRangeStartDate = self.index[splitIndex]
        testingRangeEndDate = self.index[-1]
        return testingRangeStartDate, testingRangeEndDate

    def splitDataByDate(self, splitDate, startDate=None):
        assert splitDate in list(self.index), "Date not found"
        self.splitIndex = list(self.index).index(splitDate)
        try:
            self.startPointIndex = list(self.index).index(startDate)
        except Exception:
            self.startPointIndex = 0

        testingRangeStartDate = self.index[self.splitIndex]
        testingRangeEndDate = self.index[-1]
        return testingRangeStartDate, testingRangeEndDate

    def xTrain(self):
        return self.scaledTimeSeries[
            self.startPointIndex + self.maxLookBack : self.splitIndex - 1
        ]

    def yTrain(self):
        return self.scaledTimeSeries[
            self.startPointIndex + self.maxLookBack + 1 : self.splitIndex
        ]

    def xTest(self):
        return self.scaledTimeSeries[self.splitIndex - 1 : -1]

    def yTest(self):
        return self.scaledTimeSeries[self.splitIndex + 1 :]

    def xTrainLSTM(self):
        return self.xLSTM[self.startPointIndex : self.splitIndex]

    def yTrainLSTM(self):
        return self.yLSTM[self.startPointIndex : self.splitIndex]

    def xTestLSTM(self):
        return self.xLSTM[self.splitIndex :]

    def yTestLSTM(self):
        return self.yLSTM[self.splitIndex :]

    def dataHARtrain(self):
        dataHARtrain = dict.fromkeys(self.dataHAR)
        for dataSeries in self.dataHAR:
            dataHARtrain[dataSeries] = self.dataHAR[dataSeries][
                self.startPointIndex : self.splitIndex
            ]
        return dataHARtrain

    def dataHARtest(self):
        dataHARtest = dict.fromkeys(self.dataHAR)
        for dataSeries in self.dataHAR:
            dataHARtest[dataSeries] = self.dataHAR[dataSeries][self.splitIndex :]
        return dataHARtest


class ErrorMetrics:
    def __init__(
        self, errorVectors, errorMatrices, modelName="NA", modelType="NA", testSetSize=0,
        oneDayAheadError = []):

        self.errorVector = errorVectors
        self.errorMatrix = errorMatrices
        self.modelName = modelName
        self.modelType = modelType
        self.testSetSize = testSetSize
        self.oneDayAheadError = oneDayAheadError


def createVarianceVector(data, assetList, dateIndex):

    assetList = list(set(assetList))
    assetList.sort()

    date = data.index[dateIndex]

    assetCombos = list(combinations_with_replacement(assetList, 2))
    assetCombos = [combo[0] + "-" + combo[1] for combo in assetCombos]

    varianceVector = data.loc[date][assetCombos].values

    assert not any(np.isinf(varianceVector)), "Inf"

    return np.array(varianceVector)


def covMatFromVector(varianceVector, noAssets):

    covarianceMatrix = np.zeros((noAssets, noAssets))
    covarianceMatrix.T[np.tril_indices(noAssets, 0)] = varianceVector

    ilower = np.tril_indices(noAssets, -1)
    covarianceMatrix[ilower] = covarianceMatrix.T[ilower]

    return covarianceMatrix


def varVectorFromCovMat(covarianceMatrix):

    assert covarianceMatrix.shape[0] == covarianceMatrix.shape[1]

    ilower = np.tril_indices(covarianceMatrix.shape[0], 0)
    varianceVector = covarianceMatrix[ilower]

    return varianceVector


def dataPrecprocessingOxfordMan(path, fileName, rawData=None):

    if rawData == None:
        rawData = pd.read_csv(path + fileName, skiprows=1, sep="\,", engine="python")
    rawData.columns = rawData.columns.str.replace('"', "")
    rawData.head()
    rawData["Dates"] = pd.to_datetime(rawData["Dates"], format="%Y%m%d").dt.date
    rawData = rawData.set_index("Dates")

    preprocessedData = rawData[["DJI_rv", "FTSE_rv", "GDAXI_rv", "N225_rv", "EUR_rv"]]

    preprocessedData.columns = preprocessedData.columns.str.replace("_rv", "")

    preprocessedData = preprocessedData.interpolate(limit=3).dropna()

    for i in range(0, len(preprocessedData.columns)):
        preprocessedData["Log" + preprocessedData.columns[i]] = np.log(
            preprocessedData[preprocessedData.columns[i]]
        )

    return preprocessedData


def loadScaleDataOxfordMan(assetList):

    path = "./Data/"
    fileName = "realized.library.0.1.csv"

    preprocessedData = dataPrecprocessingOxfordMan(path, fileName)

    timeSeries = np.zeros(
        np.array(preprocessedData["Log" + assetList[0]]).reshape(-1, 1).shape
    )
    for asset in assetList:
        timeSeries = np.hstack(
            (timeSeries, np.array(preprocessedData["Log" + asset]).reshape(-1, 1))
        )
    timeSeries = timeSeries[:, 1:]

    scaler = MinMaxScaler(feature_range=(0.0, 1))
    scaledTimeSeries = scaler.fit_transform(timeSeries)

    return Data(scaledTimeSeries, scaler, assetList, "OxfordMan")


def loadScaleDataMultivariate(
    assetList,
    loadpath,
    startDate=dt.datetime(1999, 1, 6),
    endDate=dt.datetime(2008, 12, 31),
):
    def getHeader(loadpath):

        header = pd.read_excel(loadpath + "no_trade.xls")
        header = header.set_index("date")
        header = header.drop(["BTI", "GSK", "ITT", "TM", "UVV"], axis=1)
        try:
            header = header.drop(["Unnamed: 0"], axis=1)
        except:
            pass

        columnNames = []
        for asset in header.columns.tolist()[:-1]:
            index = list(header.columns).index(asset)
            columnNames.append(asset + "-" + asset)
            for crossAsset in header.columns[index + 1 :].tolist():
                columnNames.append(asset + "-" + crossAsset)

        columnNames.append(header.columns[-1] + "-" + header.columns[-1])
        columnNames.sort()

        return columnNames

    noAssets = len(assetList)
    data = pd.read_csv(
        loadpath + "RVOC_6m.csv", engine="python", skiprows=[1], index_col="Var1"
    )
    data = data.drop("Unnamed: 0", axis=1)
    data.columns = getHeader(loadpath)
    data.index = pd.to_datetime(data.index, format="%Y%m%d")
    assert startDate in list(data.index), str(startDate) + " not in index"
    assert endDate in list(data.index), str(endDate) + " not in index"
    data = data[pd.Timestamp(startDate) : pd.Timestamp(endDate)]

    varVectorList = []
    for i in range(1, len(data.index)):
        varVector = createVarianceVector(data, assetList, i)
        covMat = np.real(matrixLogarithm(covMatFromVector(varVector, noAssets)))
        varVector = varVectorFromCovMat(covMat).reshape(1, -1, order="C")
        varVectorList.append(varVector)
    varVectorData = np.concatenate(varVectorList, axis=0)

    scaler = MinMaxScaler(feature_range=(0.0, 1))
    scaledTimeSeries = scaler.fit_transform(varVectorData)

    return Data(scaledTimeSeries, scaler, noAssets, "Multivariate", data.index)


def calculateErrorVectors(
    data, model, forecastHorizon, windowMode, windowSize=None, silent=True, startInd=25
):

    if model.modelType == "LSTM":
        data.createLSTMDataSet(model.lookBack)

    # When using multiprocessing, limit CPU Usage depending of model Type
    if current_process().name is not "MainProcess":
        if model.modelType == "LSTM":
            limitCPU(200)
        elif model.modelType == "ESN":
            limitCPU(200)
        elif model.modelType == "HAR":
            limitCPU(20)

    finalInd = data.scaledTimeSeries.shape[0] - forecastHorizon - data.maxLookBack
    # finalInd = data.scaledTimeSeries.shape[0] - 31
    assert windowMode.upper() in [
        "EXPANDING",
        "ROLLING",
        "FIXED",
    ], "Window Mode not recognized"

    def modelForecast(model, index):

        forecast, actual = model.multiStepAheadForecast(
            data, forecastHorizon, index, windowMode, windowSize
        )

        forecast = data.scaler.inverse_transform(forecast)
        actual = data.scaler.inverse_transform(actual)

        if data.nameDataSet == "OxfordMan":
            actual = np.exp(actual)
            forecast = np.exp(forecast)

        if data.nameDataSet == "Multivariate":
            forecast = np.array(
                [
                    matrixExponetial(covMatFromVector(vector, data.noAssets))
                    for vector in forecast
                ]
            )
            actual = np.array(
                [
                    matrixExponetial(covMatFromVector(vector, data.noAssets))
                    for vector in actual
                ]
            )

        assert actual.shape == forecast.shape

        return actual, forecast

    errorTypesList = ["RMSE", "QLIK" ,"L1Norm"]
    errorMatrices = {"RMSE": [], "QLIK": [], "L1Norm": []}
    errorOneDay = {"RMSE": [], "QLIK": [], "L1Norm": []}
    avgErrorVectors = dict.fromkeys(errorTypesList)

    def calculateForecastingError(errorType, actual, forecast):
        def RMSE(i):
            return np.matmul(
                (actual[i : i + 1].flatten() - forecast[i : i + 1].flatten()),
                (actual[i : i + 1].flatten() - forecast[i : i + 1].flatten()).T,
            )

        def QLIK(i):
            (sign, logdet) = np.linalg.slogdet(forecast[i]*10000)

            result =  logdet + np.trace(
                np.matmul(np.linalg.inv(forecast[i]*10000), actual[i]*10000)
            )
            return result

        def L1Norm(i):
            return np.linalg.norm((actual[i] - forecast[i]), ord=1)

        errorVector = [0]
        for i in range(0, forecast.shape[0]):
            try:
                errorVector.append(eval(errorType + "(i)"))
            except:
                print("Error when calculating" + errorType)

        return np.clip(errorVector[1:], a_min=None, a_max=1).reshape(-1, 1), errorVector[1]

    for index in range(startInd, finalInd):

        if silent is False and (index % 100 == 0 or index == finalInd - 1):
            print(
                model.modelName
                + " Evaluation is at index: "
                + str(index)
                + "/ "
                + str(finalInd - 1)
            )

        actual, forecast = modelForecast(model, index)

        for errorType in errorTypesList:
            oneDayError, errorVector = calculateForecastingError(errorType, actual, forecast)
            errorMatrices[errorType].append(oneDayError)
            errorOneDay[errorType].append(errorVector)

    avgErrorVectors["RMSE"] = np.sqrt(errorMatrices["RMSE"])
    for errorType in errorTypesList:
        avgErrorVectors[errorType] = np.mean(
            np.concatenate(errorMatrices[errorType], axis=1), axis=1, keepdims=True)

        # Debug Function
        def showForecast(errorType):

            avgErrorVector = avgErrorVectors[errorType]
            errorMatrix = errorMatrices[errorType]

            ax2 = plt.subplot(2, 1, 2)
            for i in range(0, len(errorMatrix)):
                shade = str(i / (len(errorMatrix) + 0.1))
                ax2.plot(np.sqrt(errorMatrix[i]), color=shade, linestyle="dotted")
            ax2.plot(avgErrorVector, color="blue", marker="x")
            ax2.set_title("Error Vectors.")

            ax3 = plt.subplot(2, 1, 3)
            ax3.set_title("Error Vector Avg. Index: " + str(index))
            ax3.plot(avgErrorVector, color="blue", marker="x")

            plt.tight_layout()
            plt.show()

    return ErrorMetrics(
        avgErrorVectors,
        errorMatrices,
        model.modelName,
        model.modelType,
        (finalInd - startInd + forecastHorizon),
        errorOneDay
    )


def limitCPU(cpuLimit):

    try:
        limitCommand = "cpulimit --pid " + str(getpid()) + " --limit " + str(cpuLimit)
        Popen(limitCommand, shell=True)
        print("CPU Limit at " + str(cpuLimit))
    except:
        print("Limiting CPU Usage failed")
