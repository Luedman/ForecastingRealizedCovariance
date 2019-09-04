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

    # Echo State Network
    hyperparameterESN = {'internalNodes': 100, 
                            'inputScaling': 1, 
                            'inputShift': 0, 
                            'spectralRadius': 0.8, 
                            'regressionLambda': 0.01, 
                            'connectivity': 0.1, 
                            'leakingRate': 0.1, 
                            'seed': 1}

    ESN = EchoStateNetworks.ESNmodel(1,1,hyperparameterESN)
    ESN.fit(data.xTrain, data.yTrain, nForgetPoints = 100)

    # LSTM Model
    LSTM = LongShortTermMemoryNetworks.LSTMmodel(loadPath = "./LSTMmodels/32-16-8-1-Model.h5")
    data.createLSTMDataSet(LSTM.forecastHorizon,LSTM.lookBack)

    # Benchmark HAR Model
    dataHAR = dataUtils.convertHAR(data.xTrain)
    HAR = HeterogeneousAutoRegressive.HARmodel()
    HAR.fit(dataHAR)

    # Evaluate Models
    errorsESN = dataUtils.calculateRSMEVector(data, ESN, daysAhead, silent = False)
    errorsHAR = dataUtils.calculateRSMEVector(data, HAR, daysAhead, silent = True)
    errorsLSTM = dataUtils.calculateRSMEVector(data, LSTM, daysAhead, silent = False)

    # Plot Errors
    def plotErrorVectors(errorsESN, errorsHAR,errorsLSTM, alpha = 0.25):

        tTestResultESN = ttest_ind(errorsESN.errorMatrixRMSE, errorsHAR.errorMatrixRMSE)
        significantPoints = list(np.where(tTestResultESN[1]<alpha)[0])

        tTestResultLSTM = ttest_ind(errorsLSTM.errorMatrixRMSE, errorsHAR.errorMatrixRMSE)
        significantPoints = list(np.where(tTestResultLSTM[1]<alpha)[0])

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(errorsESN.vectorRSME, label = "ESN", color = 'red', marker = '*', markevery = significantPoints)
        ax.plot(errorsLSTM.vectorRSME, label = "LSTM", color = 'red', marker = '*', markevery = significantPoints)
        ax.plot(errorsHAR.vectorRSME ,label = "HAR", color = 'blue')
        ax.set_title(title + ' ' + str(daysAhead) +' Days Forecasting Error - ' + str(data.xTest.shape[0]))
        ax.text(0.95, 0.01, dumps(hyperparameterESN,indent=2)[1:-1].replace('"',''),
            verticalalignment='bottom', horizontalalignment='right', transform=ax.transAxes,
            multialignment = "left")

        plt.legend()
        plt.show()
    
    plotErrorVectors(errorsESN, errorsHAR, errorsLSTM, alpha = 0.25)
    
if __name__ == "__main__":
    evaluate()