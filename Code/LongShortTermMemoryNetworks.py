# Code Appendix
# Masterthesis: Forecasting Realized Covariance with LSTM and Echo State Networks
# Author: Lukas Schreiner, 2020
#
# Hochreiter, Sepp and Schmidhuber, Jürgen, "Long short-term memory",
# Neural computation 9, 8 (1997), pp. 1735--1780.

from tensorflow import __version__ as tfVersion
import dataUtils
import numpy as np
import pandas as pd
from itertools import product as cartProduct
from time import localtime, strftime
from gc import collect

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import (
    Dense,
    LSTM,
    Dropout,
    Input,
    TimeDistributed,
    Flatten,
    Conv1D,
    MaxPooling1D,
    ConvLSTM2D,
)
from tensorflow.compat.v2.keras.layers import Bidirectional
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.callbacks import (
    TensorBoard,
    Callback,
    EarlyStopping,
    ReduceLROnPlateau,
    LearningRateScheduler,
)
from tensorflow.keras.regularizers import l1
from tensorflow.compat.v1 import disable_eager_execution

import warnings
from os.path import exists

warnings.simplefilter(action="ignore")
disable_eager_execution()


assert tfVersion == "2.0.0"

try:
    from google.colab import files as colabFiles

    runningInColab = True
    saveLogPath = (
        "/content/gdrive/My Drive/Colab Notebooks/"
        + "LSTM_Log/HyperparameterSearchLSTMsmallinclconv4.csv"
    )
    saveModelPath = "/content/gdrive/My Drive/Colab Notebooks/"
except:
    runningInColab = False
    saveLogPath = "./Data/HyperparameterSearchLSTMsmallinclconv4.csv"
    saveModelPath = "./Models/"
errorLSTM = {}


class LSTMmodel:
    def __init__(
        self,
        forecastHorizon=1,
        lookBack=32,
        architecture=[32],
        dropout=0.0,
        regularization=0.0,
        loadPath=None,
        bidirectional=True,
        convInputLayer=False,
        modelName="LSTM",
        kernelSize=1,
        filters=5,
        poolSize=2,
    ):

        np.random.seed(1)
        self.modelType = "LSTM"
        self.modelName = modelName
        self.forecastHorizon = forecastHorizon
        self.lookBack = lookBack
        self.architecture = architecture
        self.dropout = dropout
        self.regularization = l1(regularization)
        self.bidirectional = bidirectional
        self.convInputLayer = convInputLayer
        self.kernelSize = kernelSize
        self.filters = filters
        self.batchSize = 128
        self.trainingLossEvaluation = []
        self.poolSize = poolSize

        if loadPath is not None:
            self.model = load_model(loadPath)
            try:
                self.lookBack = self.model.layers[0].output_shape[0][1]
            except:
                self.lookBack = self.model.layers[0].output_shape[1]
            try:
                self.noTimeSeries = self.model.layers[0].input_shape[0][2]
            except:
                self.noTimeSeries = self.model.layers[0].input_shape[2]

            for layer in self.model.layers:
                if hasattr(layer, "rate"):
                    self.dropout = layer.rate

            if loadPath.endswith(".h5"):
                self.modelName = loadPath[9:-3]

            print("Model Loaded")

    def createModel(self, noTimeSeries=1, gpuOptmized=False):

        # Takes self.architecture, self.bidirectional as well
        # as gpuOptmized and creates a tensorflow model out of it

        self.noTimeSeries = noTimeSeries
        xTensor = inputTensor = Input(
            shape=(self.lookBack, self.noTimeSeries), batch_size=None
        )

        if self.convInputLayer is True:
            xTensor = Conv1D(
                filters=self.filters,
                kernel_size=self.kernelSize,
                activation="relu",
                padding="causal",
                input_shape=(self.lookBack, self.noTimeSeries),
            )(xTensor)
            xTensor = MaxPooling1D(pool_size=self.poolSize)(xTensor)
            xTensor = TimeDistributed(Flatten())(xTensor)

        def LSTMCell(nodes, **kwargs):
            def cpuLSTM(nodes, **kwargs):
                return LSTM(
                    nodes,
                    kernel_initializer="random_uniform",
                    bias_initializer="ones",
                    recurrent_regularizer=self.regularization,
                    stateful=False,
                    **kwargs
                )

            def gpuLSTM(nodes, **kwargs):
                return CuDNNLSTM(
                    nodes,
                    kernel_initializer="random_uniform",
                    bias_initializer="ones",
                    recurrent_regularizer=self.regularization,
                    stateful=False,
                    **kwargs
                )

            if not self.bidirectional:
                if gpuOptmized:
                    return gpuLSTM(nodes, **kwargs)
                else:
                    return cpuLSTM(nodes, **kwargs)
            if self.bidirectional:
                if gpuOptmized:
                    return Bidirectional(gpuLSTM(nodes, **kwargs))
                else:
                    return Bidirectional(cpuLSTM(nodes, **kwargs))

        for i in range(0, len(self.architecture) - 1):
            xTensor = Dropout(self.dropout)(xTensor)
            addedLSTMCell = LSTMCell(self.architecture[i], return_sequences=True)
            xTensor = addedLSTMCell(xTensor)

        xTensor = Dropout(self.dropout)(xTensor)
        addedLSTMCell = LSTMCell(self.architecture[-1], return_sequences=False)
        xTensor = addedLSTMCell(xTensor)

        xTensor = Dense(self.noTimeSeries, activation="sigmoid")(xTensor)

        self.model = Model(inputTensor, xTensor)

    def train(self, epochs, data, learningRate=0.00001, verbose=0, saveModel=True):

        # Trains the model and logs the progress of the training process

        self.model.compile(loss="mse", optimizer=Adam(lr=learningRate, clipvalue=0.5))

        tensorboard = TensorBoard(
            log_dir="./log/" + self.modelName,
            histogram_freq=0,
            write_graph=False,
            write_images=True,
        )

        reduceLearningRate = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=10,
            min_lr=0.000001,
            cooldown=50,
            verbose=int(verbose > 0),
            min_delta=0.00001,
        )

        callbacks = [overfitCallback, reduceLearningRate]

        history = self.model.fit(
            data.xTrainLSTM(),
            data.yTrainLSTM(),
            epochs=epochs,
            batch_size=self.batchSize,
            verbose=verbose,
            callbacks=callbacks,
            validation_data=(data.xTestLSTM(), data.yTestLSTM()),
        )

        if saveModel:
            print("Model trained")
            print(self.model.summary())
            self.model.save(saveModelPath + self.modelName + "-LSTM.h5")
            print("Model Saved")

        return history

    def multiStepAheadForecast(
        self, data, forecastHorizon, presentIndex, windowMode, windowSize
    ):

        # Perform recursive forecasting
        # The windowMode mode determines wheter the model is
        # refitted after each step and which data is
        # used in the subsequent training procedure

        if False and windowMode.upper() == "EXPANDING":
            data.splitData(presentIndex, startPointIndex=0)
            self.model.fit(
                data.xTrainLSTM(),
                data.yTrainLSTM(),
                epochs=10,
                batch_size=self.lookBack,
                verbose=0,
                callbacks=[overfitCallback2, resetLearningRate],
            )
            self.model.save(saveModelPath + self.modelName + "+LSTM.h5")

        elif windowMode.upper() in ["ROLLING", "EXPANDING"]:
            if presentIndex - self.batchSize * 2 > 0:
                start_index = presentIndex - self.batchSize * 2
            else:
                start_index = 0

            data.splitData(presentIndex, start_index)
            history = self.model.fit(
                data.xTrainLSTM(),
                data.yTrainLSTM(),
                epochs=250,
                batch_size=self.batchSize,
                verbose=0,
                callbacks=[overfitCallback2, resetLearningRate],
            )

            self.trainingLossEvaluation.append(history.history["loss"][-1])
            self.model.save(saveModelPath + self.modelName + "_end.h5")

        elif windowMode.upper() == "FIXED":
            data.splitData(presentIndex, startPointIndex=0)

        def recursiveMultipleDaysAheadForecast(daysAhead, forecasts):

            backlog = np.concatenate(
                [
                    data.xTrainLSTM()[-1:, forecasts.shape[1] :],
                    forecasts[:, -self.lookBack :],
                ],
                axis=1,
            )
            assert backlog.shape[1] == self.lookBack

            testPredict = self.model.predict(backlog)

            forecasts = np.concatenate(
                [forecasts, np.expand_dims(testPredict, axis=0)], axis=1
            )

            return (
                recursiveMultipleDaysAheadForecast(daysAhead + 1, forecasts)
                if (daysAhead < forecastHorizon)
                else forecasts[0]
            )

        initForecast = np.expand_dims(np.zeros((1, data.noTimeSeries)), axis=0)[:, 1:]
        forecast = recursiveMultipleDaysAheadForecast(1, initForecast)
        actual = data.yTestLSTM()[:forecastHorizon]

        return forecast, actual


class outputLogCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):

        if epoch % 100 == 0:

            print(
                "Epoch: "
                + str(epoch)
                + " Validation Loss: "
                + str(np.round(logs["val_loss"], 6))
                + " Training Loss: "
                + str(np.round(logs["loss"], 6))
            )


def searchOptimalParamters(dataPath="./Data/", gpuOptmized=True):
    print("LSTM Grid Search")

    noLayers = [1, 2]
    noNodes = [8, 16, 32]
    dropout = [0.0]
    regularization = [0.0001, 0.001]
    lookback = [32, 16]
    bidirectional = [True]
    convInputLayer = [True]
    kernelSize = [4, 6, 8, 10]
    noFilters = [4, 8, 15, 20, 30]
    poolSize = [2, 4, 6]

    hyperParameterSpace = list(
        cartProduct(
            noLayers,
            noNodes,
            dropout,
            regularization,
            lookback,
            bidirectional,
            convInputLayer,
            kernelSize,
            noFilters,
            poolSize,
        )
    )

    assetList = ["WMT", "AAPL", "ABT"]
    data = dataUtils.loadScaleDataMultivariate(assetList, dataPath)
    splitIndex = int(0.8 * len(data.scaledTimeSeries))
    data.splitData(splitIndex)

    validationErrorMin = float("inf")
    optimalParameter = None
    trainingLog = pd.DataFrame(
        columns=[
            "Time",
            "Architecture",
            "Bidirectional",
            "Validation Loss",
            "Training Loss",
            "Epochs",
            "Lookback",
            "Dropout",
            "Regularization",
            "Kernel Size",
            "No Filters",
            "Pooling Size",
        ]
    )

    totalIterations = len(hyperParameterSpace)

    for _ in range(totalIterations):

        if exists(saveLogPath):
            trainingLog = pd.read_csv(saveLogPath, sep=",", index_col=[0])

        idx = len(trainingLog)

        noLayers = hyperParameterSpace[idx][0]
        noNodes = hyperParameterSpace[idx][1]
        dropout = hyperParameterSpace[idx][2]
        regularization = hyperParameterSpace[idx][3]
        lookback = hyperParameterSpace[idx][4]
        bidirectional = hyperParameterSpace[idx][5]
        convInputLayer = hyperParameterSpace[idx][6]
        kernelSize = hyperParameterSpace[idx][7]
        filters = hyperParameterSpace[idx][8]
        poolSize = hyperParameterSpace[idx][9]
        architecture = []

        data.createLSTMDataSet(lookback)

        for _ in range(noLayers):
            architecture.append(noNodes)

        testLSTM = LSTMmodel(
            forecastHorizon=1,
            lookBack=lookback,
            architecture=architecture,
            dropout=dropout,
            regularization=regularization,
            bidirectional=bidirectional,
            convInputLayer=convInputLayer,
            kernelSize=kernelSize,
            filters=filters,
            poolSize=poolSize,
        )

        testLSTM.createModel(gpuOptmized=gpuOptmized, noTimeSeries=data.noTimeSeries)

        trainingHistory = testLSTM.train(5000, data, saveModel=False, learningRate=0.1)

        validationError = trainingHistory.history["val_loss"][-1]
        trainingError = trainingHistory.history["loss"][-1]
        noEpochs = len(trainingHistory.history["val_loss"])

        print(
            strftime("%Y-%m-%d %H:%M:%S", localtime())
            + "   "
            + str(idx)
            + " / "
            + str(totalIterations)
            + "  Epochs: "
            + str(noEpochs)
        )
        print(str(architecture) + "  " + str(np.round(validationError, 4)))

        trainingLog.loc[idx] = {
            "Time": strftime("%Y-%m-%d %H:%M:%S", localtime()),
            "Architecture": str(architecture),
            "Bidirectional": bidirectional,
            "Validation Loss": validationError,
            "Training Loss": trainingError,
            "Epochs": noEpochs,
            "Lookback": lookback,
            "Dropout": dropout,
            "Regularization": regularization,
            "Kernel Size": kernelSize,
            "No Filters": filters,
            "Pooling Size": poolSize,
        }

        trainingLog.to_csv(saveLogPath, sep=",")

        if validationError < validationErrorMin:
            validationErrorMin = validationError
            optimalParameter = hyperParameterSpace[idx]

        del testLSTM
        del trainingLog
        collect()

    print(optimalParameter)


# Output Log (Disable for GPU)
outputLog = outputLogCallback()

# Early Stopping for initial training
overfitCallback = EarlyStopping(monitor="val_loss", min_delta=0, patience=300)

# Early Stopping for rolling/expanding window evaluation
overfitCallback2 = EarlyStopping(monitor="loss", min_delta=0, patience=20)

# Reduce learning rate when a validation loss has stopped improving
reduceLearningRate = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=10,
    min_lr=0.000001,
    cooldown=10,
    verbose=0,
    min_delta=0.0001,
)


def resetlr(epoch):
    return 0.00000001


resetLearningRate = LearningRateScheduler(resetlr)
