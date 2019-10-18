# tensorboard --logdir /Users/lukas/Desktop/HSG/2-Master/4_Masterthesis/NewCode --host=127.0.0.1
import numpy as np
import pandas as pd
from itertools import product as cartProduct
from time import localtime, strftime

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, add, Input
from tensorflow.compat.v2.keras.layers import Bidirectional
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.callbacks import TensorBoard, Callback, EarlyStopping
from tensorflow.keras.regularizers import l1
from tensorflow.compat.v1 import disable_eager_execution

import warnings

warnings.simplefilter(action="ignore")
import dataUtils

from tensorflow import __version__ as tfVersion

print("TF Version: " + str(tfVersion))

try:
    from google.colab import files as colabFiles

    runningInColab = True
except:
    runningInColab = False


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
    ):

        np.random.seed(1)
        self.modelType = "LSTM"
        self.forecastHorizon = forecastHorizon
        self.lookBack = lookBack
        self.architecture = architecture
        self.dropout = dropout
        self.regularization = l1(regularization)
        self.bidirectional = bidirectional

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

            print("Model Loaded")

    def createModel(self, noTimeSeries=1, gpuOptmized=False):

        # Takes self.architecture, self.bidirectional as well
        # as gpuOptmized and creates a tensorflow model out of it

        self.noTimeSeries = noTimeSeries
        xTensor = inputTensor = Input(shape=(self.lookBack, self.noTimeSeries))

        def LSTMCell(nodes, **kwargs):
            def cpuLSTM(nodes, **kwargs):
                return LSTM(
                    nodes,
                    kernel_initializer="random_uniform",
                    bias_initializer="ones",
                    recurrent_regularizer=self.regularization,
                    **kwargs
                )

            def gpuLSTM(nodes, **kwargs):
                return CuDNNLSTM(
                    nodes,
                    kernel_initializer="random_uniform",
                    bias_initializer="ones",
                    recurrent_regularizer=self.regularization,
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

    def train(
        self,
        epochs,
        data,
        modelname="LSTM",
        learningRate=0.0001,
        batchSize=256,
        verbose=0,
        saveModel=True,
    ):

        # Trains the model and logs the progress of the training process

        self.model.compile(
            loss="mse", optimizer=Adam(lr=learningRate, clipvalue=0.5, decay=1e-12)
        )

        tensorboard = TensorBoard(
            log_dir="./Log" + modelname,
            histogram_freq=0,
            write_graph=False,
            write_images=True,
        )

        history = self.model.fit(
            data.xTrainLSTM(),
            data.yTrainLSTM(),
            epochs=epochs,
            batch_size=batchSize,
            verbose=verbose,
            callbacks=[tensorboard, overfitCallback],
            validation_data=(data.xTestLSTM(), data.yTestLSTM()),
        )

        if saveModel:
            print("Model trained")
            print(self.model.summary())
            self.model.save(modelname + "-Model.h5")
            print("Model Saved")

        return history

    def multiStepAheadForecast(
        self, data, forecastHorizon, presentIndex, windowMode, windowSize
    ):

        # Perform recursive forecasting
        # The windowMode mode determines wheter the model is
        # refitted after each step and which data is
        # used in the subsequent training procedure

        if windowMode.upper() == "EXPANDING":
            data.splitData(presentIndex, startPointIndex=0)
            self.model.fit(
                data.xTrainLSTM(),
                data.yTrainLSTM(),
                epochs=100,
                batch_size=self.lookBack,
                verbose=0,
                callbacks=[overfitCallback2],
            )
            self.model.save("32b-Model.h5")

        elif windowMode.upper() == "ROLLING":
            data.splitData(presentIndex, presentIndex - windowSize)
            self.model.fit(
                data.xTrainLSTM(),
                data.yTrainLSTM(),
                epochs=100,
                batch_size=self.lookBack,
                verbose=0,
                callbacks=[overfitCallback2],
            )
            self.model.save("32b-Model.h5")

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


def searchOptimalParamters():

    # from tensorflow.compat.v1 import disable_eager_execution
    # disable_eager_execution()

    noLayers = [1, 2, 3]
    noNodes = [8, 16, 32, 64, 128]
    dropout = [0.1, 0.25, 0.5]
    regularization = [0.0, 0.1, 0.2]
    lookback = [16, 32, 64]
    bidirectional = [True]

    hyperParameterSpace = list(
        cartProduct(noLayers, noNodes, dropout, regularization, lookback, bidirectional)
    )

    path = "Data/"
    fileName = "realized.library.0.1.csv"
    assetList = ["DJI", "FTSE", "GDAXI", "N225", "EUR"]

    data = dataUtils.loadScaleData(assetList, path, fileName)

    totalIterations = len(hyperParameterSpace)
    interationNo = 0
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
        ]
    )

    for parameterSet in hyperParameterSpace[345:]:

        noLayers = parameterSet[0]
        noNodes = parameterSet[1]
        dropout = parameterSet[2]
        regularization = parameterSet[3]
        lookback = parameterSet[4]
        bidirectional = parameterSet[5]
        architecture = []

        for layer in range(1, noLayers + 1):
            architecture.append(min(noNodes * layer, 128))

        data.createLSTMDataSet(lookback)

        testLSTM = LSTMmodel(
            forecastHorizon=1,
            lookBack=lookback,
            architecture=architecture,
            dropout=dropout,
            regularization=regularization,
            bidirectional=bidirectional,
        )

        testLSTM.createModel(gpuOptmized=True)

        trainingHistory = testLSTM.train(
            5000, data, modelname="HyperparamterTest", saveModel=False
        )

        validationError = trainingHistory.history["val_loss"][-1]
        trainingError = trainingHistory.history["loss"][-1]
        noEpochs = len(trainingHistory.history["val_loss"])

        interationNo += 1
        print(
            strftime("%Y-%m-%d %H:%M:%S", localtime())
            + "   "
            + str(interationNo)
            + " / "
            + str(totalIterations)
            + "  Epochs: "
            + str(noEpochs)
        )
        print(
            str(architecture)
            + "  "
            + str(bidirectional)
            + "  "
            + str(np.round(validationError, 4))
        )

        trainingLog.loc[interationNo] = [
            strftime("%Y-%m-%d %H:%M:%S", localtime()),
            architecture,
            bidirectional,
            validationError,
            trainingError,
            noEpochs,
            lookback,
            dropout,
            regularization,
        ]

        if runningInColab:
            saveLogPath = (
                "/content/gdrive/My Drive/Colab Notebooks/"
                + "LSTM_Log/HyperparameterSearchLSTM2.csv"
            )
        else:
            saveLogPath = "./Data/HyperparameterSearchLSTM2.csv"
        trainingLog.to_csv(saveLogPath)

        if validationError < validationErrorMin:
            validationErrorMin = validationError
            optimalParameter = parameterSet

    print(optimalParameter)


# Output Log (Disable for GPU)
outputLog = outputLogCallback()

# Early Stopping for initial training
overfitCallback = EarlyStopping(monitor="val_loss", min_delta=0.0001, patience=50)

# Early Stopping for rolling/expanding window evaluation
overfitCallback2 = EarlyStopping(monitor="loss", min_delta=0.0001, patience=5)
