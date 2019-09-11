# Project Scripts
import dataUtils
import HeterogeneousAutoRegressive
import EchoStateNetworks
import LongShortTermMemoryNetworks

# Packages
from scipy.stats import ttest_ind
import numpy as np
from matplotlib import pyplot as plt
from json import dumps



import warnings
warnings.filterwarnings("ignore")


def evaluate():

    # USER INPUT
    path        = "/Users/lukas/Desktop/HSG/2-Master/4_Masterthesis/Code/Data/Preprocessing/"
    fileName    = "realized.library.0.1.csv"
    asset       = "DJI"
    daysAhead = 30

    testSetPartition = 250
    data = dataUtils.loadScaleDataUnivariate(asset, path, fileName)
    data.xTest = data.xTest[:testSetPartition]
    data.yTest = data.yTest[:testSetPartition]
    #data.xTest = data.xTrain
    #data.yTest = data.yTrain

    # Echo State Network
    hyperparameterESN = {'internalNodes': 50, 
                            'inputScaling': 1, 
                            'inputShift': 0, 
                            'spectralRadius': 0.7, 
                            'regressionLambda': 0.01, 
                            'connectivity': 0.2, 
                            'leakingRate': 0.0, 
                            'seed': 2}

    ESN = EchoStateNetworks.ESNmodel(1,1,hyperparameterESN)
    ESN.fit(data.xTrain, data.yTrain, nForgetPoints = 50)

    # LSTM Model
    LSTM = LongShortTermMemoryNetworks.LSTMmodel(loadPath = "./LSTMmodels/32-Model.h5")
    data.createLSTMDataSet(LSTM.lookBack, LSTM.forecastHorizon)

    # Benchmark HAR Model
    dataHAR = dataUtils.convertHAR(data.xTrain)
    HAR = HeterogeneousAutoRegressive.HARmodel()
    HAR.fit(dataHAR)

    # Evaluate Models
    windowMode = "Expanding"
    windowSize = 30

    dataUtils.limitCPU(200)
    evaulationESN = dataUtils.calculateRSMEVector(data, ESN, daysAhead, 
                                                windowMode = windowMode,
                                                windowSize = windowSize,
                                                silent = False)
    evaluationLSTM = dataUtils.calculateRSMEVector(data, LSTM, daysAhead,  
                                                windowMode = "Rolling",
                                                windowSize = windowSize,
                                                silent = False)
    evaluationHAR = dataUtils.calculateRSMEVector(data, HAR, daysAhead,  
                                                windowMode = windowMode,
                                                windowSize = windowSize,
                                                silent = False)

    # Plot Errors
    def plotErrorVectors(evaulationESN, evaluationHAR, evaluationLSTM, errorType, alpha = 0.25):

        tTestResultESN = ttest_ind(evaulationESN.errorMatrix[errorType], evaluationHAR.errorMatrix[errorType])
        significantPointsESN = list(np.where(tTestResultESN[1]<alpha)[0])

        tTestResultLSTM = ttest_ind(evaluationLSTM.errorMatrix[errorType], evaluationHAR.errorMatrix[errorType])
        significantPointsLSTM = list(np.where(tTestResultLSTM[1]<alpha)[0])
            
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(evaulationESN.errorVector[errorType], label = "ESN", color = 'red', marker = '*', markevery = significantPointsESN)
        ax.plot(evaluationLSTM.errorVector[errorType], label = "LSTM", color = 'green', marker = '*', markevery = significantPointsLSTM)
        ax.plot(evaluationHAR.errorVector[errorType] ,label = "HAR", color = 'blue')
        ax.set_title(errorType + " " + str(daysAhead) +' Days Forecasting Error \n Test Set Size:' + str(data.xTest.shape[0])+ "  " + windowMode + " Window")
        ax.text(0.95, 0.01, dumps(hyperparameterESN,indent=2)[1:-1].replace('"',''),
            verticalalignment='bottom', horizontalalignment='right', transform=ax.transAxes,
            multialignment = "left")

        plt.legend()
        plt.show()
    
    plotErrorVectors(evaulationESN, evaluationHAR, evaluationLSTM, errorType = "RMSE", alpha = 0.25)
    plotErrorVectors(evaulationESN, evaluationHAR, evaluationLSTM, errorType = "QLIK", alpha = 0.25)
    plotErrorVectors(evaulationESN, evaluationHAR, evaluationLSTM, errorType = "L1Norm", alpha = 0.25)


if __name__ == "__main__":
    evaluate()