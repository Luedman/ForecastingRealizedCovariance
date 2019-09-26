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

from sklearn.metrics import mean_squared_error
from scipy.stats import ttest_ind
import warnings

# Project Scripts
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
        reservoirStateResult = - self.leakingRate * prevReservoirState + reservoirStateResult

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
            self.collectedStateMatrixTraining = collectedStateMatrix
            self.networkTrained = True
        except:
            self.reservoirReadout = np.ones((yTrain.shape[1], self.internalNodes))
            self.networkTrained = False
            print("Failed to train Network")

        assert self.reservoirReadout.shape == (yTrain.shape[1], self.internalNodes)

        outputSequence = self.__outputActivationFunction(np.matmul(self.reservoirReadout, collectedStateMatrix)).T
        outputSequence = (outputSequence - np.ones((outputSequence.shape))* self.inputShift) / self.inputScaling
        
        def showOutputSequence(start = 0, end = -1):
            #Debug Function
            actual = xTrain[nForgetPoints:]
            plt.plot(actual[start:end], label = "Actual")
            plt.plot(outputSequence[start:end], label = "ESN Fit")
            plt.legend()
            plt.show()

        self.modelResidualMatrix = yTrain[nForgetPoints:] - outputSequence
        
        if False:
            global counter
            print(datetime.now().strftime("%d.%b %Y %H:%M:%S") + " " + str(counter) + " ESN Trained")
            counter += 1 

        return

    def evaluate(self, data, showPlot = False):

        assert data.xTest.shape[1] == data.yTest.shape[1], "X and Y should be of same lenght (shape[1])"

        collectedStateMatrix = self.__collectStateMatrix(data.xTest,100)

        output = self.test(data.xTest, collectedStateMatrix)
        
        if data.scaler is not None:
            data.yTest = data.scaler.inverse_transform(data.yTest)
            output = data.scaler.inverse_transform(output)

        yHat = np.exp(output)
        data.yTest = np.exp(data.yTest)

        try:
            rmse = np.sqrt(mean_squared_error(data.yTest[-yHat.shape[0]:], yHat))
        except:
            rmse = float('inf')
            print("Error when calculating RSME")

        if showPlot:
            plt.plot(np.exp(data.yTest[-yHat.shape[0]:]), label = "Var")
            plt.plot(yHat, label = "ESN")
            plt.legend() 
            plt.show()

        return rmse, yHat
    '''
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
    '''
    def multiStepAheadForecast(self, data, noStepsAhead, startIndex, 
                                    windowMode = "Expanding", windowSize = 400):
        
        noSamples = 1
        assert self.networkTrained == True, "ESN is not trained of failed in the Process"
        assert windowMode.upper() in ["EXPANDING", "ROLLING", "FIXED"], "Window Mode not recognized"

        if windowMode.upper() == "EXPANDING":
            newxTrain = np.concatenate([data.xTrain, data.xTest[:startIndex]])
            newyTrain = np.concatenate([data.yTrain, data.yTest[:startIndex]])
            self.fit(newxTrain[:], newyTrain[:], 50)
        elif windowMode.upper() == "ROLLING":
            newxTrain = np.concatenate([data.xTrain, data.xTest[:startIndex]])
            newyTrain = np.concatenate([data.yTrain, data.yTest[:startIndex]])
            self.fit(newxTrain[-windowSize:], newyTrain[-windowSize:], 50)
        else: pass

        randomStartIndices = np.random.randint(0, self.modelResidualMatrix.shape[0] + 1 - noStepsAhead, size = noSamples)
        randomResidualsMatrix = np.array([self.modelResidualMatrix[randomIndex:randomIndex + noStepsAhead,:] for randomIndex in randomStartIndices])

        prevReservoirState = self.collectedStateMatrixTraining[:,-1].reshape(-1,1)

        forecastVector = np.zeros((noStepsAhead,1))
        forecastVector[-1] = data.xTest[startIndex]

        for i in range(0,randomResidualsMatrix.shape[1]):

            oneStepForecastingSamples = []

            for residualVector in randomResidualsMatrix:

                reservoirState = self.__reservoirState(forecastVector[i-1], prevReservoirState)
                #  + residualVector[i],
                
                oneStepForecast = self.__outputActivationFunction(np.matmul(self.reservoirReadout, prevReservoirState))
                oneStepForecast = (oneStepForecast - self.inputShift)/self.inputScaling

                if np.absolute(oneStepForecast) < 1.01:
                    oneStepForecastingSamples.append(oneStepForecast)
            
            if oneStepForecastingSamples:
                forecastVector[i] = np.average(oneStepForecastingSamples)
                prevReservoirState = reservoirState
            else:
                #print("Error when forecasting")
                forecastVector[i] = 0
                prevReservoirState = reservoirState
        
        def showForecast():
            # Debug Function
            ax1 = plt.subplot(2, 1, 1)
            ax1.set_title("Actual vs Forecast")
            ax1.plot(forecastVector, label = "ESN")
            ax1.plot(data.xTest[startIndex: startIndex + noStepsAhead], label = "actual")
            ax1.legend()
            ax2 = plt.subplot(2, 1, 2)
            ax2.set_title("Reservoir Readout")
            ax2.bar(list(range(0,self.internalNodes)),self.reservoirReadout.reshape(-1))
            plt.tight_layout()
            plt.show()

        return forecastVector

def hedgeAlgorithm():
    pass


def searchOptimalParamters():
    print("Hyperparameter Search")

    # Disable Warnings (especially overflow)
    warnings.filterwarnings("ignore")

    # USER INPUT Specifiy Data Path
    path        = "/Users/lukas/Desktop/HSG/2-Master/4_Masterthesis/Code/Data/Preprocessing/"
    fileName    = "realized.library.0.1.csv"
    asset       = "DJI"
    daysAhead = 10

    windowMode = "Expanding"
    windowSize = 0

    testSetPartition = 250
    data = dataUtils.loadScaleDataUnivariate(asset, path, fileName, scaleData = True)
    data.xTest = data.xTest[:testSetPartition]
    data.yTest = data.yTest[:testSetPartition]

    internalNodesRange  = [100]
    shiftRange          = [0]
    scalingRange        = [1]
    specRadRange        = [0.05, 0.1,0.2,0.4,0.6, 0.8]
    regLambdaRange      = [0.01, 0.001,1e-4, 1e-6 ,1e-8, 1e-10]
    connectivityRange   = [0.1]
    leakingRate         = [0, 0.1]
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
    avgErrorMinRMSE = float('inf')
    avgErrorMinQLIK = float('inf')

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
        testEsn.fit(data.xTrain, data.yTrain, 100)

        evaulationESN = dataUtils.calculateErrorVectors(data, testEsn, daysAhead, 
                                                windowMode = windowMode,
                                                windowSize = windowSize,
                                                silent = False)                                     
        #weights = np.array(range(daysAhead, 0, -1)).reshape(-1,1)

        avgRMSE = np.average(evaulationESN.errorVector["RMSE"])
        if avgRMSE < avgErrorMinRMSE:
            avgErrorMinRMSE = avgRMSE
            optimalParametersRMSE = hyperparameterESN

        avgQLIK = np.average(evaulationESN.errorVector["QLIK"])
        if  avgQLIK < avgErrorMinQLIK:
            avgErrorMinQLIK = avgQLIK
            optimalParametersQLIK = hyperparameterESN
        
        interationNo += 1
        print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
        print(interationNo, " / ", totalIterations, "RMSE: " + str(avgRMSE) + "  QLIK: " + str(avgQLIK))
        print("Min RMSE: " + str(avgErrorMinRMSE) + "  Min QLIK: " + str(avgErrorMinQLIK))
        print(hyperparameterESN)
    
    print("Hyperparameter Search Finished: ")
    print("Optimal Paramenters RMSE: ")
    print(optimalParametersRMSE)
    print("Optimal Paramenters QLIK: ")
    print(optimalParametersQLIK)
    return


def baysianOptimization():
    print("Baysian Optimization")

    # Disable Warnings (especially overflow)
    warnings.filterwarnings("ignore")

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

    def esnEvaluation( spectralRadius,
                            regressionLambda, 
                            connectivity, 
                            leakingRate):
        
        hyperparameterESN = {'internalNodes': 100, 
                                'inputScaling': 1, 
                                'inputShift': 0, 
                                'spectralRadius': spectralRadius, 
                                'regressionLambda': regressionLambda, 
                                'connectivity': connectivity, 
                                'leakingRate': leakingRate, 
                                'seed': 1}

        ESN = ESNmodel(1,1,hyperparameterESN)
        ESN.fit(data.xTrain, data.yTrain, nForgetPoints = 50)

        evaluationESN = dataUtils.calculateErrorVectors(data, ESN, daysAhead, 
                                                    silent = True, 
                                                    windowMode = "Fixed",
                                                    windowSize = 400)

        return np.average(evaluationESN.errorVector["RMSE"])*-1

    pbounds = { 'spectralRadius':   (0.1, 1),
                'regressionLambda': (1, 1e-10),
                'connectivity':     (0.01, 0.1),
                'leakingRate' :     (0,0.2)}

    dataUtils.limitCPU(100)
    optimizer = BayesianOptimization(f=esnEvaluation,
                                        pbounds=pbounds,
                                        random_state=1)

    optimizer.maximize(init_points=50, n_iter=50)
    print(optimizer.max)

    # {'target': -5.107364346276386e-05, 'params': {'connectivity': 0.25, 'leakingRate': 0.3045764247831867, 'regressionLambda': 0.03349326637933621, 'spectralRadius': 0.28370282580070894}}
    # 200 {'target': -5.817736792645783e-05, 'params': {'connectivity': 0.39209320009591453, 'leakingRate': 0.027453634574218677, 'regressionLambda': 0.018121378307801052, 'spectralRadius': 0.11798213612570352}}
    # 400 {'target': -5.766103436439476e-05, 'params': {'connectivity': 0.02351810808542145, 'leakingRate': 0.10729748881666834, 'regressionLambda': 0.05968872229967288, 'spectralRadius': 0.1932189099820555}}
    
if __name__ == "__main__":
    #pass
    baysianOptimization()
    #searchOptimalParamters()
    #evaluateESN()

    