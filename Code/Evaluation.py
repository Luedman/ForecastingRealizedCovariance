# Notes:
# Better Names for LSTM1
# forget points == lookback

# Project Scripts
import dataUtils
import HeterogeneousAutoRegressive
import EchoStateNetworks
import LongShortTermMemoryNetworks
import errorFunctions
import Hedging

# Packages
from scipy.stats import ttest_ind
import numpy as np
from json import dumps
from time import time
from sklearn.metrics import mean_squared_error
from copy import deepcopy
import datetime as dt
from pandas import Timestamp
from matplotlib import pyplot as plt
import pickle
import warnings

warnings.filterwarnings("ignore")

try:
    from google.colab import files as colabFiles

    runningInColab = True
except:
    runningInColab = False
    dataUtils.limitCPU(150)

assetList = ["WMT", "AAPL", "ABT"]


def evaluate(
    assetList,
    startDate=dt.datetime(1999, 1, 6),
    endDate=dt.datetime(2008, 12, 31),
    splitDate=None,
    dataLoadpath="./Data/",
    modelLoadpath="./Models/",
    saveName="evaluation200601-200806"):
    
    data = dataUtils.loadScaleDataMultivariate(
        assetList, dataLoadpath, startDate=startDate, endDate=endDate
    )
    daysAhead = 30
    trainingFraction = 0.95
    splitIndex = int(list(data.index).index(Timestamp(splitDate)))

    hyperparameterESNUnivariate = {
        "internalNodes": 200,
        "inputScaling": 1,
        "inputShift": 0,
        "spectralRadius": 0.3,
        "regressionLambda": 0.02,
        "connectivity": 0.4,
        "leakingRate": 0.03,
        "seed": 1,
    }

    esnParameterSet = {
        "internalNodes": (40, 80, 120),
        "spectralRadius": (0.2, 0.3, 0.4),
        "regressionLambda": (1e-6, 1e-5, 1e-4),
        "connectivity": (0.25, 0.1, 0.05),
        "leakingRate": (0.01, 0.01, 0.1),
    }


    def initESN(name, n, esnParameterSet = esnParameterSet):
        return EchoStateNetworks.ESNmodel(
            nInputNodes=data.noTimeSeries,
            nOutputNodes=data.noTimeSeries,
            hyperparameter={
                key: esnParameterSet[key][n - 1] for key in esnParameterSet
            },
            modelName=name,
        )

    ESNtest = EchoStateNetworks.ESNmodel(nInputNodes=data.noTimeSeries,
                                        nOutputNodes=data.noTimeSeries,
                                        hyperparameter={'connectivity': 0.06, 
                                                        'internalNodes': 217.2,
                                                        'leakingRate': 0.09, 
                                                        'regressionLambda': 0.001, 
                                                        'spectralRadius': 0.48},
                                        modelName="TestESN")

    HAR = HeterogeneousAutoRegressive.HARmodel()

    ESN1 = initESN("ESN1", 1, esnParameterSet)
    ESN2 = initESN("ESN2", 2, esnParameterSet)
    ESN3 = initESN("ESN3", 3, esnParameterSet)

    ESNExperts = Hedging.HedgingAlgorithm(
        [deepcopy(ESN1), deepcopy(ESN2), deepcopy(ESN3), deepcopy(ESNtest)],
        modelName="EchoStateExperts",
        updateRate=2,
    )

    ESN1e = initESN("ESNexpert1", 1)
    ESN2e = initESN("ESNexpert2", 2)
    ESN3e = initESN("ESNexpert3", 3)
    ESN4e = initESN("ESNexpert4", 2)

    LSTM1 = LongShortTermMemoryNetworks.LSTMmodel(modelName="LSTM1",
        loadPath=modelLoadpath + "1-LSTM.h5")
    LSTM2 = LongShortTermMemoryNetworks.LSTMmodel(modelName="LSTM2",
        loadPath=modelLoadpath + "2-LSTM.h5")
    LSTM3 = LongShortTermMemoryNetworks.LSTMmodel(modelName="LSTM3",
        loadPath=modelLoadpath + "3-LSTM.h5")

    LSTM1e = LongShortTermMemoryNetworks.LSTMmodel(
        loadPath=modelLoadpath + "1-LSTM.h5")
    LSTM2e = LongShortTermMemoryNetworks.LSTMmodel(
        loadPath=modelLoadpath + "2-LSTM.h5")
    LSTM3e = LongShortTermMemoryNetworks.LSTMmodel(
        loadPath=modelLoadpath + "3-LSTM.h5")

    LSTM4e = LongShortTermMemoryNetworks.LSTMmodel(
        loadPath=modelLoadpath + "3-LSTM.h5"
    )

    LSTMExperts = Hedging.HedgingAlgorithm(
        [LSTM1e, LSTM2e, LSTM3e], modelName="LSTMExperts", updateRate=2
    )

    HybridExpert = Hedging.HedgingAlgorithm(
        [LSTM4e, ESN4e], modelName="HybridExperts", updateRate=2
    )

    data.createLSTMDataSet(LSTM1.lookBack)
    data.createHARDataSet()
    testingRangeStartDate, testingRangeEndDate = data.splitData(splitIndex)

    HAR.fit(data.dataHARtrain())

    modelList = [HAR, HybridExpert, ESN1, ESN2, ESN3 ,ESNExperts, LSTM1, LSTM2, LSTM3, LSTMExperts]

    # Evaluate Models
    windowMode = "Expanding"
    windowSize = 64
    silent = False

    start = time()
    evalResults = []


    for model in modelList:
        evaluationModel = dataUtils.calculateErrorVectors(
            data,
            model,
            daysAhead,
            windowMode=windowMode,
            windowSize=windowSize,
            silent=silent,
            startInd=splitIndex)

        evalResults.append(evaluationModel)

    end = time()
    print("Evaluation Time: %6.2f Minutes" % ((end - start) / 60))
    
    with open(str(saveName) + '.pkl', 'wb') as f:
        pickle.dump([evalResults, 
                    testingRangeStartDate, 
                    testingRangeEndDate, 
                    assetList, 
                    data], f)
    
    for errorType in ["RMSE", "QLIK", "L1Norm"]:
        errorFunctions.plotErrorVectors(
            evalResults,
            errorType,
            testingRangeStartDate,
            testingRangeEndDate,
            assetList,
            data
        )

    print("Done")
    return evalResults

def loadfromSaved(saveName: str, show_only_models: list):

    with open(str(saveName) + '.pkl', 'rb') as f:
        loadedVars = pickle.load(f)
    
    evalResults = loadedVars[0]
    evalResultsFiltered = []
    for i in show_only_models:
        evalResultsFiltered.append(evalResults[i])
    testingRangeStartDate = loadedVars[1]
    testingRangeEndDate = loadedVars[2] 
    assetList = loadedVars[3]
    data = loadedVars[4]

    for errorType in ["RMSE", "QLIK", "L1Norm"]:
        errorFunctions.plotErrorVectors(
            evalResultsFiltered,
            errorType,
            testingRangeStartDate,
            testingRangeEndDate,
            assetList,
            data
        )

    return evalResultsFiltered


if __name__ == "__main__":
    #assetList = ["WMT", "AAPL", "ABT"]
    #splitDate = dt.datetime(2006, 1, 3)
    #endDate = dt.datetime(2008, 6, 30)
    #evaluate(assetList, splitDate=splitDate, endDate=endDate)
    loadfromSaved(saveName="evaluation200601-200806", show_only_models=[0,1,5,9])

