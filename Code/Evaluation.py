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
    #path        = "/Users/lukas/Desktop/HSG/2-Master/4_Masterthesis/Code/Data/Preprocessing/"
    path        = ""
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
    hyperparameterESN = {'internalNodes': 100, 
                            'inputScaling': 1, 
                            'inputShift': 0, 
                            'spectralRadius': 0.17, 
                            'regressionLambda': 1.0, 
                            'connectivity': 0.011, 
                            'leakingRate': 0.08, 
                            'seed': 1}

    ESN = EchoStateNetworks.ESNmodel(1,1,hyperparameterESN)
    ESN.fit(data.xTrain, data.yTrain, nForgetPoints = 50)

    # LSTM Model
    #loadPath = "./LSTMmodels/32-Model.h5"
    loadPath = "32-Model.h5"

    LSTM = LongShortTermMemoryNetworks.LSTMmodel(loadPath = loadPath)
    data.createLSTMDataSet(LSTM.lookBack, LSTM.forecastHorizon)

    # Benchmark HAR Model
    dataHAR = dataUtils.convertHAR(data.xTrain)
    HAR = HeterogeneousAutoRegressive.HARmodel()
    HAR.fit(dataHAR)

    # Evaluate Models
    windowMode = "Fixed"
    windowSize = 0
    silent = False

    try:
        # Try Multiprocessing
        print("Multiprocessing")
        from multiprocessing import Pool, set_start_method, current_process
        set_start_method('spawn', True)

        pool = Pool(processes = 3)
        input_list = [[data, ESN, daysAhead, windowMode, windowSize, silent],
                      [data, LSTM, daysAhead, "Fixed", 30, silent],
                      [data, HAR, daysAhead, windowMode, windowSize, silent]]
        results = pool.starmap(dataUtils.calculateErrorVectors, input_list)
        pool.close()

        evaluationESN = results[0]
        evaluationLSTM = results[1]
        evaluationHAR = results[2]

    except:
        # Sequential Alternative
        print("Multiprocessing failed")
        dataUtils.limitCPU(200)
        evaluationHAR = dataUtils.calculateErrorVectors(data, HAR, daysAhead,  
                                                    windowMode = windowMode,
                                                    windowSize = windowSize,
                                                    silent = False)

        evaluationESN = dataUtils.calculateErrorVectors(data, ESN, daysAhead, 
                                                    windowMode = windowMode,
                                                    windowSize = windowSize,
                                                    silent = False)

        evaluationLSTM = dataUtils.calculateErrorVectors(data, LSTM, daysAhead,  
                                                windowMode = "Rolling",
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
    
    # Plot Charts
    for errorType in ["RMSE", "QLIK", "L1Norm"]:
        plotErrorVectors(evaluationESN, evaluationHAR, evaluationLSTM, errorType = errorType, alpha = 0.25)


if __name__ == "__main__":
    evaluate()