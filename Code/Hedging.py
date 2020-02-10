# Code Appendix
# Masterthesis: Forecasting Realized Covariance with LSTM and Echo State Networks
# Author: Lukas Schreiner, 2020
#
# Freund, Yoav and Schapire, Robert E, "A decision-theoretic generalization of on-line learning and an application to boosting", 
# Journal of computer and system sciences 55, 1 (1997), pp. 119--139.
# Freund, Yoav and Schapire, Robert E, "Adaptive game playing using multiplicative weights", 
# Games and Economic Behavior 29, 1-2 (1999), pp. 79--103.

import numpy as np
import EchoStateNetworks, LongShortTermMemoryNetworks
from sklearn.metrics import mean_squared_error

class HedgingAlgorithm:
    def __init__(self, modelList, modelName, updateRate=1):

        self.modelList = modelList
        self.noModels = len(modelList)
        self.modelType = modelName
        self.modelName = modelName
        self.updateRate = updateRate

    def multiStepAheadForecast(self, *args, **kwargs):

        modelWeights = np.array([1 / self.noModels for model in self.modelList])

        modelForecasts = []
        for model in self.modelList:
            forecast, actual = model.multiStepAheadForecast(*args, **kwargs)
            modelForecasts.append(forecast)

        expertForecast = np.zeros((forecast.shape))
        for i in range(forecast.shape[0]):

            for m in range(self.noModels):
                expertForecast[i] = np.add(
                    expertForecast[i], modelForecasts[m][i] * modelWeights[m]
                )
                loss = np.sqrt(mean_squared_error(actual[i], modelForecasts[m][i]))
                modelWeights[m] *= np.exp(-self.updateRate * loss)

            modelWeights = [weights / sum(modelWeights) for weights in modelWeights]
            assert np.round(sum(modelWeights), 2) == 1.0

        return expertForecast, actual
