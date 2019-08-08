# ToDo: 
# Use Tanh
#
#

import numpy as np
import pandas as pd
import scipy as sc

from matplotlib import pyplot as plt
from time import gmtime, strftime
from itertools import product as cartProduct
from datetime import datetime
from copy import copy
from json import dumps

from sklearn.metrics import mean_squared_error
import dataUtils
import HeterogeneousAutoRegressive

np.random.seed(1)
counter = 0

class ESNmodel:

    def __init__(self, nInputNodes, nOutputNodes, hyperparameter):

        hyperparameterESN = copy(hyperparameter)
        
        np.random.seed(1)
        self.modelType = "ESN"
        self.nInputNodes    = nInputNodes
        self.nOutputNodes   = nOutputNodes
        
        # Default Paramenters
        self.internalNodes      = 100
        self.regressionLambda   = 1e-8
        self.spectralRadius     = 1
        self.leakingRate        = 1
        self.connectivity       = min([10/self.internalNodes, 1])
        self.inputMask          = np.ones([self.internalNodes, nInputNodes])

        self.inputScaling       = np.ones((nInputNodes,1))
        self.inputShift         = np.zeros((nInputNodes, 1))

        self.networkTrained     = False
        self.reservoirMatrix    = None
        self.networkTrained     = False

        # Set optional Parameters
        if ('internalNodes' in hyperparameterESN):
            self.internalNodes      = int(hyperparameterESN['internalNodes'])
            self.connectivity       = min([10/self.internalNodes, 1])
            self.inputMask          = np.ones([self.internalNodes, nInputNodes])
            hyperparameterESN.pop('internalNodes')

        if ('seed' in hyperparameterESN):
            np.random.seed(int(hyperparameterESN['seed']))
            hyperparameterESN.pop('seed')

        if ('regressionLambda' in hyperparameterESN):
            self.regressionLambda = hyperparameterESN['regressionLambda']
            hyperparameterESN.pop('regressionLambda')

        if ('spectralRadius' in hyperparameterESN):
            self.spectralRadius = hyperparameterESN['spectralRadius']
            hyperparameterESN.pop('spectralRadius')

        if ('leakingRate' in hyperparameterESN):
            self.leakingRate = hyperparameterESN['leakingRate']
            hyperparameterESN.pop('leakingRate')

        if ('connectivity' in hyperparameterESN):
            self.connectivity = hyperparameterESN['connectivity']
            hyperparameterESN.pop('connectivity')

        if ('leakingRate' in hyperparameterESN):
            self.leakingRate = hyperparameterESN['leakingRate']
            hyperparameterESN.pop('leakingRate')

        if ('inputMask' in hyperparameterESN):
            self.inputMask = hyperparameterESN['inputMask']
            hyperparameterESN.pop('inputMask')

        if ('inputScaling' in hyperparameterESN):
            self.inputScaling = np.ones(self.nInputNodes) * hyperparameterESN['inputScaling']
            hyperparameterESN.pop('inputScaling')

        if ('inputShift' in hyperparameterESN):
            self.inputShift = np.ones([self.nInputNodes, 1])* hyperparameterESN['inputShift']
            hyperparameterESN.pop('inputShift')

        # Check if all input arguments were used
        assert bool(hyperparameterESN) is False, \
            "Init: Input Argument not recognized. This Option does not exist"
        
        success = 0
        while success == 0:
            try: 
                rvs = sc.stats.norm(loc=0, scale=1).rvs
                internalWeights = sc.sparse.random(self.internalNodes, self.internalNodes, density=self.connectivity, data_rvs=rvs).A
                eigs = sc.sparse.linalg.eigs(internalWeights, 1,  which = 'LM')
                
                maxVal = max(abs(eigs[1]))
                internalWeights = internalWeights / (1.25 * maxVal)
                success = 1
            except:
                success = 0

        internalWeights *= self.spectralRadius

        assert internalWeights.shape == (self.internalNodes, self.internalNodes)

        self.reservoirMatrix = internalWeights

    @staticmethod
    def __activationFunction(input_vector, function = "Sigmoid"):
        
        def sigmoidActivation(x):
            return 1. / (1. + np.exp(-x))
        
        def tanhActivation(x):
            return np.tanh(x)
        
        if function.upper() == "SIGMOID":
            result = np.array(list(map(sigmoidActivation,np.array(input_vector))))
        elif function.upper() == "TANH":
            result = np.array(list(map(tanhActivation,np.array(input_vector))))
        else:
            raise NameError('Argument "function" for __activationFunction not found.')
          
        return result

    @staticmethod
    def __outputActivationFunction(inputVector):

        result = np.array(inputVector)
        return result

    @staticmethod
    def __scaleInput(self, inputVector):
        pass


    def __reservoirState(self, prevOutput, prevReservoirState):

        prevReservoirState = prevReservoirState.reshape(self.internalNodes, self.nInputNodes)

        activation = np.matmul(self.reservoirMatrix, prevReservoirState) + \
            self.inputScaling * np.matmul(self.inputMask, prevOutput).reshape(self.internalNodes, self.nInputNodes) + self.inputShift

        reservoirStateResult = self.__activationFunction(activation,"Sigmoid")

        reservoirStateResult = (1 - self.leakingRate) * prevReservoirState + self.leakingRate * reservoirStateResult

        assert reservoirStateResult.shape == (self.internalNodes, self.nInputNodes)

        return reservoirStateResult

    def __collectStateMatrix(self,inputVector, nForgetPoints):

        collectedStateMatrix = np.zeros([self.internalNodes, 1])

        for i in range(0, len(inputVector)):

            collectedStateMatrix = np.concatenate((collectedStateMatrix, \
                self.__reservoirState(inputVector[i], collectedStateMatrix[:,-1])), axis = 1)
        
        return collectedStateMatrix[:, nForgetPoints + 1:]

    def test(self, xTest, collectedStateMatrix):

        assert self.networkTrained == True, "Network isn't trained yet"

        outputSequence = self.__outputActivationFunction(np.matmul(self.reservoirReadout, collectedStateMatrix))
        outputSequence = (outputSequence - np.ones((outputSequence.shape)) * self.inputShift) / self.inputScaling 

        return outputSequence.T
    
    def fit(self, xTrain, yTrain, nForgetPoints):

        collectedStateMatrix = self.__collectStateMatrix(xTrain, nForgetPoints)

        gamma = np.matmul(collectedStateMatrix, collectedStateMatrix.T) + self.regressionLambda * np.eye(self.internalNodes)

        cov = np.matmul(collectedStateMatrix, yTrain[nForgetPoints:])
        
        try:
            self.reservoirReadout = np.matmul(np.linalg.inv(gamma), cov).T
            self.networkTrained = True
        except:
            self.reservoirReadout = np.ones((yTrain.shape[1], self.internalNodes))
            self.networkTrained = False
            print("Failed to train Network")

        assert self.reservoirReadout.shape == (yTrain.shape[1], self.internalNodes)

        outputSequence = self.__outputActivationFunction(np.matmul(self.reservoirReadout, collectedStateMatrix)).T
        outputSequence = (outputSequence - np.ones((outputSequence.shape))* self.inputShift) / self.inputScaling

        self.modelResidualMatrix = yTrain[nForgetPoints:] - outputSequence
        
        if False:
            global counter
            print(datetime.now().strftime("%d.%b %Y %H:%M:%S") + " " + str(counter) + " ESN Trained")
            counter += 1 

        return

    def evaluate(self, xTest, yTest, scaler, showPlot = False):

        assert xTest.shape[1] == yTest.shape[1], "X and Y should be of same lenght (shape[1])"

        collectedStateMatrix = self.__collectStateMatrix(xTest,100)

        output = self.test(xTest, collectedStateMatrix)
        
        if scaler is not None:
            yTest = scaler.inverse_transform(yTest)
            output = scaler.inverse_transform(output)

        yHat = np.exp(output)
        yTest = np.exp(yTest)

        try:
            rmse = np.sqrt(mean_squared_error(yTest[-yHat.shape[0]:], yHat))
        except:
            rmse = float('inf')
            print("Error when calculating RSME")

        if showPlot:
            plt.plot(np.exp(yTest[-yHat.shape[0]:]), label = "Var")
            plt.plot(yHat, label = "ESN")
            plt.legend() 
            plt.show()

        return rmse, yHat
    
    def oneStepAheadForecast(self, xTest, yTest):

        assert self.networkTrained == True, "Network isn't trained yet"

        noSamples = 20

        collectedStateMatrix = self.__collectStateMatrix(xTest, 0)

        randomIndices = np.random.randint(0,self.modelResidualMatrix.shape[0] + 1, size = noSamples)
        randomResiduals = self.modelResidualMatrix[randomIndices]

        startIndex = 3000

        forecasts = []
        
        for residual in randomResiduals:

            reservoirState = self.__reservoirState(xTest[startIndex] + residual, collectedStateMatrix[:,startIndex -1])

            forecasts.append(self.__outputActivationFunction(np.matmul(self.reservoirReadout, reservoirState)).T)

        oneStepAheadForecast = np.average(forecasts)

        return oneStepAheadForecast

    def multiStepAheadForecast(self,xTest, noStepsAhead, startIndex):

        noSamples = 2

        randomStartIndices = np.random.randint(0, self.modelResidualMatrix.shape[0] + 1 - noStepsAhead, size = noSamples)
        randomResidualsMatrix = np.array([self.modelResidualMatrix[randomIndex:randomIndex + noStepsAhead,:] for randomIndex in randomStartIndices])

        collectedStateMatrix = self.__collectStateMatrix(xTest, 0)
        prevReservoirState = collectedStateMatrix[:,startIndex -1]

        forecastVector = np.zeros((noStepsAhead,1))
        forecastVector[-1] = xTest[startIndex]

        for i in range(0,randomResidualsMatrix.shape[1]):

            for residualVector in randomResidualsMatrix:

                reservoirState = self.__reservoirState(forecastVector[i-1] + residualVector[i], prevReservoirState)

                oneStepForecastingSamples = np.matmul(self.reservoirReadout, prevReservoirState)

            forecastVector[i] = np.average(oneStepForecastingSamples)
            prevReservoirState = reservoirState

        multiStepAheadForecast = forecastVector
        
        def showForecast():
            # Debug Function
            ax1 = plt.subplot(2, 1, 1)
            ax1.set_title("Actual vs Forecast")
            ax1.plot(multiStepAheadForecast, label = "ESN")
            ax1.plot(xTest[startIndex: startIndex + noStepsAhead], label = "actual")
            ax1.legend()
            ax2 = plt.subplot(2, 1, 2)
            ax2.set_title("Reservoir Readout")
            ax2.bar(list(range(0,self.internalNodes)),self.reservoirReadout.reshape(-1))
            plt.tight_layout()
            plt.show()

        return multiStepAheadForecast


def searchOptimalParamters():
    print("Hyperparameter Search")

    # USER INPUT Specifiy Data Path
    path        = "/Users/lukas/Desktop/HSG/2-Master/4_Masterthesis/Code/Data/Preprocessing/"
    fileName    = "realized.library.0.1.csv"
    asset       = "DJI"

    xTrain, yTrain, xTest, yTest, scaler = dataUtils.loadScaleDataUnivariate(asset, path, fileName, scaleData = True)

    internalNodesRange  = [100]
    shiftRange          = [0]
    scalingRange        = [1]
    specRadRange        = list(np.round(np.linspace(0.1,1.1,num = 20),4))
    regLambdaRange      = [0.01, 1e-4, 1e-6 ,1e-8,1e-10, 1e-12]
    connectivityRange   = [0.1,0.2,0.3,0.05]
    leakingRate         = list(np.round(np.linspace(0.0,1.0,num = 5),2))
    seed                = [1]

    hyperParameterSpace = list(cartProduct(internalNodesRange, 
                                                    scalingRange, 
                                                    shiftRange, 
                                                    specRadRange,
                                                    regLambdaRange,
                                                    connectivityRange,
                                                    leakingRate,
                                                    seed))

    totalIterations = len(hyperParameterSpace)
    interationNo = 0
    minRMSE = float('inf')

    for parameterSet in hyperParameterSpace:

        hyperparameterESN = {'internalNodes':    parameterSet[0],
                            'inputScaling':      parameterSet[1],
                            'inputShift':        parameterSet[2],
                            'spectralRadius':    parameterSet[3],
                            'regressionLambda':  parameterSet[4],
                            'connectivity':      parameterSet[5],
                            'leakingRate':       parameterSet[6],
                            'seed':              parameterSet[7]}
        
        testEsn = ESNmodel(1,1,hyperparameterESN)
        
        testEsn.fit(xTrain, yTrain, 100)
        #rsmeTestRun, _ = testEsn.evaluate(xTest, yTest, scaler)
        rsmeTestRun = np.average(dataUtils.calculateRSMEVector(xTest[:250], 
                                                                yTest[:250], 
                                                                testEsn, 
                                                                10, scaler))

        if rsmeTestRun < minRMSE:
            minRMSE = rsmeTestRun
            optimalParameters = hyperparameterESN
        
        interationNo += 1
        print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
        print(interationNo, " / ", totalIterations, " MinRSME: ", minRMSE, " RSME: ", rsmeTestRun)
        print(hyperparameterESN)
    
    print(optimalParameters)
    return optimalParameters

def evaluateESN():
    print("Evaluate ESN")

    # USER INPUT Specifiy Data Path
    path        = "/Users/lukas/Desktop/HSG/2-Master/4_Masterthesis/Code/Data/Preprocessing/"
    fileName    = "realized.library.0.1.csv"
    asset       = "DJI"

    xTrain, yTrain, xTest, yTest, scaler = dataUtils.loadScaleDataUnivariate(asset, path, fileName)

    hyperparameterESN = {'internalNodes':       100, 
                            'inputScaling':     1, 
                            'inputShift':       0, 
                            'spectralRadius':   0.1, 
                            'regressionLambda': 1e-10, 
                            'connectivity':     0.3, 
                            'seed':             1}

    testEsn = ESNmodel(1,1,hyperparameterESN)

    testEsn.fit(xTrain, yTrain, 100)

    dataHAR = dataUtils.convertHAR(xTrain)
    HAR = HeterogeneousAutoRegressive.HARmodel()
    HAR.fit(dataHAR, silent=True)

    rsmeTestRun, _ = testEsn.evaluate(xTest, yTest, scaler)
    print("RSEM Test Run: " + str(rsmeTestRun))

    esnRSME, esnRSME2 = dataUtils.calculateRSMEVector(xTest[:250], yTest[:250], testEsn, 30, 
                        scaler, RSMETimeAxis = True,
                        xTrain = xTrain, 
                        yTrain = yTrain)

    esnHAR, esnHAR2 = dataUtils.calculateRSMEVector(xTest[:250], yTest[:250], HAR, 30, scaler, RSMETimeAxis = True)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(esnRSME, label = "ESN")
    ax.plot(esnHAR, label = "HAR")
    ax.set_title('Echo State HAR RSME Forecasting Error')
    ax.text(0.95, 0.01, dumps(hyperparameterESN,indent=2),
        verticalalignment='bottom', horizontalalignment='right', transform=ax.transAxes,
        multialignment = "left")

    plt.legend()
    plt.show()

    

if __name__ == "__main__":
    searchOptimalParamters()
    #evaluateESN()

    