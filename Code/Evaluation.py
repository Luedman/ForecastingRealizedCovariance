# Code Appendix
# Masterthesis: Forecasting Realized Covariance with LSTM and Echo State Networks
# Author: Lukas Schreiner, 2020

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
    dataUtils.limitCPU(250)



def evaluate(
    assetList,
    startDate=dt.datetime(1999, 1, 6),
    endDate=dt.datetime(2008, 12, 31),
    splitDate=None,
    dataLoadpath="./Data/",
    modelLoadpath="./Models/",
    saveName="evaluation200601-200806_3Assets",
):

    # Load Data
    data = dataUtils.loadScaleDataMultivariate(
        assetList, dataLoadpath, startDate=startDate, endDate=endDate
    )
    daysAhead = 30
    trainingFraction = 0.95
    splitIndex = int(list(data.index).index(Timestamp(splitDate)))

    esnParameterSetUnivariate = {
        "internalNodes": (40, 80, 120),
        "spectralRadius": (0.5, 0.3, 0.1),
        "regressionLambda": (1e-3, 1e-3, 1e-5),
        "connectivity": (5/40, 5/80, 10/120),
        "leakingRate": (0.3, 0.2, 0.2)
    }

    esnParameterSet = {
        "internalNodes": (65, 70, 75),
        "spectralRadius": (0.30, 0.35, 0.3),
        "regressionLambda": (5e-5, 5e-5, 5e-5),
        "connectivity": (10/80, 10/70, 10/75),
        "leakingRate": (0.1, 0.2, 0.25)
    }

    # initialize the ESNs
    def initESN(name, n, esnParameterSet=esnParameterSet):
        return EchoStateNetworks.ESNmodel(
            nInputNodes=data.noTimeSeries,
            nOutputNodes=data.noTimeSeries,
            hyperparameter={
                key: esnParameterSet[key][n - 1] for key in esnParameterSet
            },
            modelName=name,
        )

    HAR = HeterogeneousAutoRegressive.HARmodel()

    ESN1 = initESN("ESN1", 1, esnParameterSetUnivariate)
    ESN2 = initESN("ESN2", 2, esnParameterSetUnivariate)
    ESN3 = initESN("ESN3", 3, esnParameterSetUnivariate)

    ESNExperts = Hedging.HedgingAlgorithm(
        [deepcopy(ESN1), deepcopy(ESN2), deepcopy(ESN3)],
        modelName="EchoStateExperts",
        updateRate=2,
    )

    LSTM1 = LongShortTermMemoryNetworks.LSTMmodel(
        modelName="LSTM1", loadPath=modelLoadpath + "1-Asset-Model-1-LSTM.h5"
    )
    LSTM2 = LongShortTermMemoryNetworks.LSTMmodel(
        modelName="LSTM2", loadPath=modelLoadpath + "1-Asset-Model-2-LSTM.h5"
    )
    LSTM3 = LongShortTermMemoryNetworks.LSTMmodel(
        modelName="LSTM3", loadPath=modelLoadpath + "1-Asset-Model-3-LSTM.h5"
    )

    LSTM1e = LongShortTermMemoryNetworks.LSTMmodel(loadPath=modelLoadpath + "1-Asset-Model-1-LSTM.h5")
    LSTM2e = LongShortTermMemoryNetworks.LSTMmodel(loadPath=modelLoadpath + "1-Asset-Model-2-LSTM.h5")
    LSTM3e = LongShortTermMemoryNetworks.LSTMmodel(loadPath=modelLoadpath + "1-Asset-Model-3-LSTM.h5")

    LSTMExperts = Hedging.HedgingAlgorithm(
        [LSTM1e, LSTM2e, LSTM3e], modelName="LSTMExperts", updateRate=2
    )

    LSTM4e = LongShortTermMemoryNetworks.LSTMmodel(loadPath=modelLoadpath + "1-Asset-Model-2-LSTM.h5")

    ESN4e = initESN("ESN4", 2, esnParameterSet)

    HybridExpert = Hedging.HedgingAlgorithm(
        [LSTM4e, ESN4e], modelName="HybridExperts", updateRate=2
    )

    data.createLSTMDataSet(LSTM1.lookBack)
    data.createHARDataSet()
    testingRangeStartDate, testingRangeEndDate = data.splitData(splitIndex)

    HAR.fit(data.dataHARtrain())

    modelList = [HAR, ESN1, ESN2, ESN3, ESNExperts, HybridExpert, LSTM1, LSTM2, LSTM3, LSTMExperts]

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
            startInd=splitIndex,
        )

        evalResults.append(evaluationModel)

    end = time()
    print("Evaluation Time: %6.2f Minutes" % ((end - start) / 60))

    with open(str(saveName) + ".pkl", "wb") as f:
        pickle.dump(
            [evalResults, testingRangeStartDate, testingRangeEndDate, assetList, data],
            f,
        )

    for errorType in ["RMSE", "QLIK", "L1Norm"]:
        errorFunctions.plotErrorVectors(
            evalResults,
            errorType,
            testingRangeStartDate,
            testingRangeEndDate,
            assetList,
            data,
        )

    print("Done")
    return evalResults


def loadfromSaved(saveName: str, show_only_models: list = None):

    with open(str(saveName) + ".pkl", "rb") as f:
        loadedVars = pickle.load(f)

    evalResults = loadedVars[0]

    if show_only_models == None:
        show_only_models = range(len(evalResults))

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
            data,
        )

    return evalResultsFiltered


if __name__ == "__main__":
    #assetList = ["WMT","AAPL","ABT"]
    assetList = ["AAPL"]
    splitDate = dt.datetime(2006, 1, 3)
    endDate = dt.datetime(2008, 12, 31)
    evaluate(assetList, splitDate=splitDate, endDate=endDate)
    #loadfromSaved(saveName="evaluation200601-20086_3Assets")