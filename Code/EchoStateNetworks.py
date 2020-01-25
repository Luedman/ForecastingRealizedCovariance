# Python Packages
import numpy as np
import pandas as pd
import scipy as sc

from matplotlib import pyplot as plt
from time import gmtime, strftime
from itertools import product as cartProduct
from datetime import datetime
from copy import copy
from json import dumps
from bayes_opt import BayesianOptimization
import datetime as dt

from sklearn.metrics import mean_squared_error
from scipy.stats import ttest_ind
import warnings

# Project Scripts
import dataUtils
import HeterogeneousAutoRegressive

import warnings

warnings.filterwarnings("ignore")
np.random.seed(1)
counter = 0


class ESNmodel:
    def __init__(self, nInputNodes, nOutputNodes, hyperparameter, modelName="ESN"):

        hyperparameterESN = copy(hyperparameter)

        np.random.seed(1)
        self.modelName = modelName
        self.modelType = "ESN"
        self.nInputNodes = nInputNodes
        self.nOutputNodes = nOutputNodes

        # Default Paramenters
        self.internalNodes = 100
        self.regressionLambda = 1e-8
        self.spectralRadius = 1
        self.leakingRate = 1
        self.connectivity = min([10 / self.internalNodes, 1])
        self.inputMask = np.ones([self.internalNodes, nInputNodes])

        self.inputScaling = 1
        self.inputShift = 0

        self.networkTrained = False
        self.reservoirMatrix = None

        # Set optional Parameters
        if "internalNodes" in hyperparameterESN:
            self.internalNodes = int(hyperparameterESN["internalNodes"])
            self.connectivity = min([10 / self.internalNodes, 1])
            self.inputMask = np.ones([self.internalNodes, nInputNodes])
            hyperparameterESN.pop("internalNodes")

        if "seed" in hyperparameterESN:
            np.random.seed(int(hyperparameterESN["seed"]))
            hyperparameterESN.pop("seed")

        if "regressionLambda" in hyperparameterESN:
            self.regressionLambda = hyperparameterESN["regressionLambda"]
            hyperparameterESN.pop("regressionLambda")

        if "spectralRadius" in hyperparameterESN:
            self.spectralRadius = hyperparameterESN["spectralRadius"]
            hyperparameterESN.pop("spectralRadius")

        if "leakingRate" in hyperparameterESN:
            self.leakingRate = hyperparameterESN["leakingRate"]
            hyperparameterESN.pop("leakingRate")

        if "connectivity" in hyperparameterESN:
            self.connectivity = hyperparameterESN["connectivity"]
            hyperparameterESN.pop("connectivity")

        if "leakingRate" in hyperparameterESN:
            self.leakingRate = hyperparameterESN["leakingRate"]
            hyperparameterESN.pop("leakingRate")

        if "inputMask" in hyperparameterESN:
            self.inputMask = hyperparameterESN["inputMask"]
            hyperparameterESN.pop("inputMask")

        if "inputScaling" in hyperparameterESN:
            self.inputScaling = hyperparameterESN["inputScaling"]
            hyperparameterESN.pop("inputScaling")

        if "inputShift" in hyperparameterESN:
            self.inputShift = hyperparameterESN["inputShift"]
            hyperparameterESN.pop("inputShift")

        # Check if all input arguments were used
        assert (
            bool(hyperparameterESN) is False
        ), "Init: Input Argument not recognized. This Option does not exist"

        success = 0
        while success == 0:
            try:
                rvs = sc.stats.norm(loc=0, scale=1).rvs
                internalWeights = sc.sparse.random(
                    self.internalNodes,
                    self.internalNodes,
                    density=self.connectivity,
                    data_rvs=rvs,
                ).A
                eigs = sc.sparse.linalg.eigs(internalWeights, 1, which="LM")

                maxVal = max(abs(eigs[1]))
                internalWeights = internalWeights / (1.25 * maxVal)
                success = 1
            except:
                success = 0

        internalWeights *= self.spectralRadius

        assert internalWeights.shape == (self.internalNodes, self.internalNodes)

        self.reservoirMatrix = internalWeights
        self.collectedStateMatrix = np.zeros([self.internalNodes, 1])

    @staticmethod
    def __activationFunction(input_vector, function="Sigmoid"):
        def sigmoidActivation(x):
            return 1.0 / (1.0 + np.exp(-x))

        def tanhActivation(x):
            return np.tanh(x)

        if function.upper() == "SIGMOID":
            result = np.array(list(map(sigmoidActivation, np.array(input_vector))))
        elif function.upper() == "TANH":
            result = np.array(list(map(tanhActivation, np.array(input_vector))))
        else:
            raise NameError('Argument "function" for __activationFunction not found.')

        return result

    @staticmethod
    def __outputActivationFunction(inputVector):

        result = np.array(inputVector)
        return result

    def __reservoirState(self, prevOutput, prevReservoirState):

        prevReservoirState = prevReservoirState.reshape(self.internalNodes, 1)

        activation = (
            np.matmul(self.reservoirMatrix, prevReservoirState)
            + self.inputScaling
            * np.matmul(self.inputMask, prevOutput).reshape(self.internalNodes, 1)
            + self.inputShift
        )

        reservoirStateResult = self.__activationFunction(activation, "Sigmoid")
        reservoirStateResult = (
            -self.leakingRate * prevReservoirState + reservoirStateResult
        )

        assert reservoirStateResult.shape == (self.internalNodes, 1)

        return reservoirStateResult

    def __collectStateMatrix(self, inputVector, nForgetPoints):

        for i in range(self.collectedStateMatrix.shape[1] - 1, inputVector.shape[0]):

            self.collectedStateMatrix = np.concatenate(
                (
                    self.collectedStateMatrix,
                    self.__reservoirState(
                        inputVector[i], self.collectedStateMatrix[:, -1]
                    ),
                ),
                axis=1,
            )

        return self.collectedStateMatrix[:, nForgetPoints + 1 :]

    def test(self, xTrain, nForgetPoints=100):

        assert self.networkTrained == True, "Network isn't trained yet"

        collectedStateMatrix = self.__collectStateMatrix(xTrain, nForgetPoints)

        outputSequence = self.__outputActivationFunction(
            np.matmul(self.reservoirReadout, collectedStateMatrix)
        )
        outputSequence = (
            outputSequence - np.ones((outputSequence.shape)) * self.inputShift
        ) / self.inputScaling

        avgRMSE = np.mean(np.power(xTrain[nForgetPoints:] - outputSequence.T, 2))

        return avgRMSE

    def fit(self, data, nForgetPoints):

        xTrain, yTrain = data.xTrain(), data.yTrain()

        collectedStateMatrix = self.__collectStateMatrix(xTrain, nForgetPoints)

        gamma = np.matmul(
            collectedStateMatrix, collectedStateMatrix.T
        ) + self.regressionLambda * np.eye(self.internalNodes)

        cov = np.matmul(collectedStateMatrix, yTrain[nForgetPoints:])

        try:
            self.reservoirReadout = np.matmul(np.linalg.inv(gamma), cov).T
            self.collectedStateMatrixTraining = collectedStateMatrix
            self.networkTrained = True
        except:
            self.reservoirReadout = np.ones((yTrain.shape[1], self.internalNodes))
            self.networkTrained = False
            print("Failed to train Network")

        assert self.reservoirReadout.shape == (yTrain.shape[1], self.internalNodes)

        outputSequence = self.__outputActivationFunction(
            np.matmul(self.reservoirReadout, collectedStateMatrix)
        ).T
        outputSequence = (
            outputSequence - np.ones((outputSequence.shape)) * self.inputShift
        ) / self.inputScaling

        self.modelResidualMatrix = yTrain[nForgetPoints:] - outputSequence

        if False:
            global counter
            print(
                datetime.now().strftime("%d.%b %Y %H:%M:%S")
                + " "
                + str(counter)
                + " ESN Trained"
            )
            counter += 1

        return

    def evaluate(self, data, showPlot=False):

        assert (
            data.xTest.shape[1] == data.yTest.shape[1]
        ), "X and Y should be of same lenght (shape[1])"

        collectedStateMatrix = self.__collectStateMatrix(data.xTest, 100)

        output = self.test(data.xTest, collectedStateMatrix)

        if data.scaler is not None:
            data.yTest = data.scaler.inverse_transform(data.yTest)
            output = data.scaler.inverse_transform(output)

        yHat = np.exp(output)
        data.yTest = np.exp(data.yTest)

        try:
            rmse = np.sqrt(mean_squared_error(data.yTest[-yHat.shape[0] :], yHat))
        except:
            rmse = float("inf")
            print("Error when calculating RSME")

        if showPlot:
            plt.plot(np.exp(data.yTest[-yHat.shape[0] :]), label="Var")
            plt.plot(yHat, label="ESN")
            plt.legend()
            plt.show()

        return rmse, yHat

    def multiStepAheadForecast(
        self, data, forecastHorizon, index, windowMode, windowSize, noSamples=1
    ):

        if windowMode.upper() == "EXPANDING":
            data.splitData(index, startPointIndex=0)
            self.fit(data, nForgetPoints=50)
        elif windowMode.upper() == "ROLLING":
            data.splitData(index, index - windowSize)
            self.fit(data, nForgetPoints=50)
        elif windowMode.upper() == "FIXED":
            data.splitData(index, startPointIndex=0)
            assert self.networkTrained == True

        actual = data.yTest()[:forecastHorizon]

        randomStartIndices = np.random.randint(
            0, self.modelResidualMatrix.shape[0] + 1 - forecastHorizon, size=noSamples
        )
        randomResidualsMatrix = np.array(
            [
                self.modelResidualMatrix[randomIndex : randomIndex + forecastHorizon, :]
                for randomIndex in randomStartIndices
            ]
        )

        prevReservoirState = self.collectedStateMatrixTraining[:, -1].reshape(-1, 1)

        multiStepForecast = np.zeros((forecastHorizon, self.nOutputNodes))
        multiStepForecast[-1] = data.yTest()[0]

        for i in range(0, randomResidualsMatrix.shape[1]):

            oneStepForecastingSamples = []

            for residualVector in randomResidualsMatrix:

                reservoirState = self.__reservoirState(
                    multiStepForecast[i - 1] + residualVector[i], prevReservoirState
                )

                oneStepForecast = self.__outputActivationFunction(
                    np.matmul(self.reservoirReadout, prevReservoirState)
                )
                oneStepForecast = (
                    oneStepForecast - self.inputShift
                ) / self.inputScaling

                if all(np.absolute(oneStepForecast)) < 1.01:
                    oneStepForecastingSamples.append(oneStepForecast)

            if oneStepForecastingSamples:
                multiStepForecast[i] = np.average(oneStepForecastingSamples)
                prevReservoirState = reservoirState
            else:
                multiStepForecast[i] = multiStepForecast[i - 1]
                prevReservoirState = reservoirState

        def showForecast():
            # Debug Function
            ax1 = plt.subplot(2, 1, 1)
            ax1.set_title("Actual vs Forecast")
            ax1.plot(multiStepForecast, label="ESN")
            ax1.plot(data.xTest()[index : index + forecastHorizon], label="actual")
            ax1.legend()
            ax2 = plt.subplot(2, 1, 2)
            ax2.set_title("Reservoir Readout")
            ax2.bar(
                list(range(0, self.internalNodes)), self.reservoirReadout.reshape(-1)
            )
            plt.tight_layout()
            plt.show()

        return multiStepForecast, actual

# Hyperparameter Grid Search
def searchOptimalParamters():
    print("Hyperparameter Search ESN")

    # Disable Warnings (especially overflow)
    warnings.filterwarnings("ignore")

    # USER INPUT Specifiy Data Path
    path = "./Data/"
    fileName = "realized.library.0.1.csv"
    assetList = ["DJI"]
    daysAhead = 10

    windowMode = "Expanding"
    windowSize = 0

    testSetPartition = 250
    data = dataUtils.loadScaleData(assetList, path, fileName, scaleData=True)
    data.xTest = data.xTest[:testSetPartition]
    data.yTest = data.yTest[:testSetPartition]

    internalNodesRange = [100]
    shiftRange = [0]
    scalingRange = [1]
    specRadRange = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8]
    regLambdaRange = [0.01, 0.001, 1e-4, 1e-6, 1e-8, 1e-10]
    connectivityRange = [0.1]
    leakingRate = [0, 0.1]
    seed = [1]

    hyperParameterSpace = list(
        cartProduct(
            internalNodesRange,
            scalingRange,
            shiftRange,
            specRadRange,
            regLambdaRange,
            connectivityRange,
            leakingRate,
            seed,
        )
    )

    totalIterations = len(hyperParameterSpace)
    interationNo = 0
    avgErrorMinRMSE = float("inf")
    avgErrorMinQLIK = float("inf")

    for parameterSet in hyperParameterSpace:

        hyperparameterESN = {
            "internalNodes": parameterSet[0],
            "inputScaling": parameterSet[1],
            "inputShift": parameterSet[2],
            "spectralRadius": parameterSet[3],
            "regressionLambda": parameterSet[4],
            "connectivity": parameterSet[5],
            "leakingRate": parameterSet[6],
            "seed": parameterSet[7],
        }

        testEsn = ESNmodel(1, 1, hyperparameterESN)
        testEsn.fit(data.xTrain, data.yTrain, 100)

        evaulationESN = dataUtils.calculateErrorVectors(
            data,
            testEsn,
            daysAhead,
            windowMode=windowMode,
            windowSize=windowSize,
            silent=False,
        )
        # weights = np.array(range(daysAhead, 0, -1)).reshape(-1,1)

        avgRMSE = np.average(evaulationESN.errorVector["RMSE"])
        if avgRMSE < avgErrorMinRMSE:
            avgErrorMinRMSE = avgRMSE
            optimalParametersRMSE = hyperparameterESN

        avgQLIK = np.average(evaulationESN.errorVector["QLIK"])
        if avgQLIK < avgErrorMinQLIK:
            avgErrorMinQLIK = avgQLIK
            optimalParametersQLIK = hyperparameterESN

        interationNo += 1
        print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
        print(
            interationNo,
            " / ",
            totalIterations,
            "RMSE: " + str(avgRMSE) + "  QLIK: " + str(avgQLIK),
        )
        print(
            "Min RMSE: " + str(avgErrorMinRMSE) + "  Min QLIK: " + str(avgErrorMinQLIK)
        )
        print(hyperparameterESN)

    print("Hyperparameter Search Finished: ")
    print("Optimal Paramenters RMSE: ")
    print(optimalParametersRMSE)
    print("Optimal Paramenters QLIK: ")
    print(optimalParametersQLIK)

    return

# Hyperparameter Bayesian Search
def bayesianOptimization(dataPath="./Data/"):
    print("Bayesian Optimization")

    # Disable Warnings (especially overflow)
    warnings.filterwarnings("ignore")

    # USER INPUT
    assetList = ["WMT", "AAPL", "ABT"]

    daysAhead = 30
    trainingFraction = 0.8

    data = dataUtils.loadScaleDataMultivariate(assetList, dataPath, endDate=dt.datetime(2005, 6, 1))
    splitIndex = int(trainingFraction * len(data.scaledTimeSeries))
    data.splitData(splitIndex)

    def esnEvaluation(
        internalNodes, spectralRadius, regressionLambda, connectivity, leakingRate
    ):

        hyperparameterESN = {
            "internalNodes": internalNodes,
            "inputScaling": 1,
            "inputShift": 0,
            "spectralRadius": spectralRadius,
            "regressionLambda": regressionLambda,
            "connectivity": connectivity,
            "leakingRate": leakingRate,
            "seed": 1,
        }

        ESN = ESNmodel(
            nInputNodes=data.noTimeSeries,
            nOutputNodes=data.noTimeSeries,
            hyperparameter=hyperparameterESN,
        )

        error = np.average(
            dataUtils.calculateErrorVectors(
                data,
                ESN,
                daysAhead,
                windowMode="Expanding",
                windowSize=0,
                silent=True,
                startInd=splitIndex,
            ).errorVector["RMSE"]
        )

        del ESN
        return error * -1

    pbounds = {
        "internalNodes": (50, 300),
        "spectralRadius": (0.1, 1.2),
        "regressionLambda": (0.001, 1e-12),
        "connectivity": (0.01, 0.1),
        "leakingRate": (0, 0.2),
    }

    # dataUtils.limitCPU(200)
    optimizer = BayesianOptimization(f=esnEvaluation, pbounds=pbounds, random_state=1)

    optimizer.maximize(init_points=1000, n_iter=1000)
    print(optimizer.max)

    # {'target': -5.107364346276386e-05, 'params': {'connectivity': 0.25, 'leakingRate': 0.3045764247831867, 'regressionLambda': 0.03349326637933621, 'spectralRadius': 0.28370282580070894}}
    # 200 {'target': -5.817736792645783e-05, 'params': {'connectivity': 0.39209320009591453, 'leakingRate': 0.027453634574218677, 'regressionLambda': 0.018121378307801052, 'spectralRadius': 0.11798213612570352}}
    # 400 {'target': -5.766103436439476e-05, 'params': {'connectivity': 0.02351810808542145, 'leakingRate': 0.10729748881666834, 'regressionLambda': 0.05968872229967288, 'spectralRadius': 0.1932189099820555}}
    # 100 {'target': -5.342171818354836e-05, 'params': {'connectivity': 0.017684561647952367, 'leakingRate': 0.14181898970022982, 'regressionLambda': 1.0, 'spectralRadius': 0.11527163386472704}}
    #
    # Multivariate
    # {'target': -0.008708270035450504,
    # 'params': {'connectivity': 0.025835148873894348, 'internalNodes': 72.03451490347274, 'leakingRate': 0.03648895092077236, 'regressionLambda': 1.0, 'spectralRadius': 0.9995766070981673}}
    # 'params': {'connectivity': 0.04772750629629654, 'internalNodes': 557.6190153055048, 'leakingRate': 0.04089044994630349, 'regressionLambda': 0.12188256360993266, 'spectralRadius': 0.12464883387813355}}
    # In Sample 20/20
    # {'target': -1.2637431314970473e-06, 'params': {'connectivity': 0.02488187774052395, 'internalNodes': 189.1262870594051, 'leakingRate': 0.06955317194910131, 'regressionLambda': 0.2491878968645953, 'spectralRadius': 0.7533981868154063}}
    # {'target': -1.2589387144155041e-06, 'params': {'connectivity': 0.030967684645691843, 'internalNodes': 171.06577934281688, 'leakingRate': 0.07757212881283436, 'regressionLambda': 0.13645814544143486, 'spectralRadius': 0.7724094784634662}}
    # {'target': -1.2178894471454817e-06, 'params': {'connectivity': 0.054962575361099754, 'internalNodes': 232.14641699364907, 'leakingRate': 0.0416388876817583, 'regressionLambda': 0.007519664416475726, 'spectralRadius': 0.24258359374681834}}
    """
        esnParameterSetInSample = {
        "internalNodes": (220, 250, 300),
        "spectralRadius": (0.65, 0.6, 0.7),
        "regressionLambda": (0.0005, 0.0005, 0.0005),
        "connectivity": (0.045, 0.04, 0.03),
        "leakingRate": (0.05, 0.05, 0.05),
    }
    """
if __name__ == "__main__":
    # pass
    bayesianOptimization()
    # searchOptimalParamters()
    # evaluateESN()
