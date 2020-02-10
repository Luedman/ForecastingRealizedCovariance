# Code Appendix
# Masterthesis: Forecasting Realized Covariance with LSTM and Echo State Networks
# Author: Lukas Schreiner, 2020
#
# The code in this file is inspired by an ESN implementation in MATLAB by H. Jaeger
# Jaeger, Herbert, "The echo state approach to analysing and training recurrent neural networks-with an erratum note", 
# Bonn, Germany: German National Research Center for Information Technology GMD Technical Report 148, 34 (2001), pp. 13.

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

        # Create reservoir matrix
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
        # Calculate the reservoir state for at given t

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
        # Calculate a series of reservoir states and store in collected sate matrix

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

    def fit(self, data, nForgetPoints):
        # Fit the ESN to the training data

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

    def test(self, xTrain, nForgetPoints=100):
        # Check wheter the ESN fits the training data well

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

    def evaluate(self, data, showPlot=False):
        # Evaluate the output of ESN.test()

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
        # Refit the ESN and create a multi step ahead forecast

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

def bayesianOptimization(dataPath="./Data/"):
    # Hyperparameter Bayesian Search
    print("Bayesian Optimization")

    # Disable Warnings (especially overflow)
    warnings.filterwarnings("ignore")

    # USER INPUT
    assetList = ["WMT", "AAPL", "ABT"]

    daysAhead = 30
    trainingFraction = 0.8

    data = dataUtils.loadScaleDataMultivariate(
        assetList, dataPath, endDate=dt.datetime(2008, 12, 31)
    )
    splitIndex = int(list(data.index).index(dt.datetime(2006, 1, 3)))
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
        "connectivity": (0.01, 0.5),
        "leakingRate": (0, 0.5),
    }
    optimizer = BayesianOptimization(f=esnEvaluation, pbounds=pbounds, random_state=1)

    optimizer.maximize(init_points=1000, n_iter=1000)
    print(optimizer.max)


if __name__ == "__main__":
    pass
    # uncomment the function needed in oder to run. 

    bayesianOptimization()
    # gridSearch()
    # evaluateESN()
