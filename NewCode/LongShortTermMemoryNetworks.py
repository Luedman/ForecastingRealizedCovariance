import numpy as np

from keras.optimizers       import Adam
from keras.models           import Sequential, load_model
from keras.layers           import Dense
from keras.layers           import LSTM
from keras.callbacks        import TensorBoard, Callback
from keras.callbacks        import EarlyStopping
from keras.regularizers     import l1
import dataUtils


class LSTMmodel:

    def __init__(self, forecastHorizon = 30, 
                                lookBack = 32, 
                                architecture = [32,32], 
                                dropout = 0.0,
                                regularization = 0.0,
                                loadPath = None):

        np.random.seed(1)
        self.modelType = "LSTM"
        self.forecastHorizon = forecastHorizon
        self.lookBack = lookBack
        self.architecture = architecture
        self.dropout = dropout
        self.regularization = l1(regularization)

        if loadPath is not None:
            self.model = load_model(loadPath)
            self.forecastHorizon = self.model.layers[-1].output_shape[1]
            self.lookBack = self.model.layers[0].output_shape[1]

            #print("Model Loaded")
            #print(self.model.summary())
 
    def createModel(self, hybridHAR = False):

        inputDimension = 1
        if hybridHAR:
            inputDimension = 3

        LSTMmodelStructure = Sequential()
        LSTMmodelStructure.add(LSTM(self.architecture[0], 
                                activation='relu',
                                return_sequences = True, 
                                input_shape = (self.lookBack, inputDimension),
                                dropout = self.dropout,
                                kernel_initializer='random_uniform',
                                bias_initializer='ones',
                                recurrent_regularizer = self.regularization))
        
        for i in range(len(self.architecture) - 1):
            LSTMmodelStructure.add(LSTM(self.architecture[i + 1], return_sequences = True, \
                dropout = self.dropout,
                kernel_initializer='random_uniform', 
                bias_initializer='ones',
                recurrent_regularizer = self.regularization))
        
        LSTMmodelStructure.add(LSTM(self.architecture [-1]))
        LSTMmodelStructure.add(Dense(self.forecastHorizon, activation='softmax'))
        
        self.model = LSTMmodelStructure

    def train(self, epochs, xTrainLSTM, yTrainLSTM, 
                        xTestLSTM, yTestLSTM, modelname = "LSTM", 
                        learningRate = 0.00001, decay=1e-6 ,batchSize = 256):

        self.model.compile(loss = 'mse', optimizer = Adam(lr=learningRate, clipvalue = 0.5))
    
        tensorboard = TensorBoard(log_dir='./Log' + modelname, 
                                histogram_freq = 0, 
                                write_graph = False, 
                                write_images = True)
        
        outputLog = outputLogCallback()

        overfitCallback = EarlyStopping(monitor = 'val_loss', 
                                        min_delta = 0, 
                                        patience = 100)
        
        self.model.fit(xTrainLSTM, yTrainLSTM, 
                            epochs = epochs, 
                            batch_size = batchSize, 
                            verbose = 0, 
                            callbacks=[tensorboard, outputLog, overfitCallback], 
                            validation_data = (xTestLSTM, yTestLSTM))

        print("Model trained")
        print(self.model.summary())
        
        self.model.save(modelname +"-Model.h5")
        print("Model Saved")

    def multiStepAheadForecast(self, xTestLSTM, forecastHorizon, startIndex):

        assert forecastHorizon == self.forecastHorizon or self.forecastHorizon == 1, \
            "Model has diffrent forecast Horizon"

        def recursiveMultipleDaysAheadForecast(daysAhead, forecasts, startIndex):

            backlog = np.append(xTestLSTM[startIndex], forecasts)[-self.lookBack:].reshape(1,self.lookBack,1)

            assert backlog.shape[1] == self.lookBack

            testPredict = self.model.predict(backlog)

            forecasts.append(testPredict[0])

            return recursiveMultipleDaysAheadForecast(daysAhead + 1, forecasts, startIndex)\
                 if (daysAhead < forecastHorizon) else forecasts

        if self.forecastHorizon == 1:
            forecast = np.array(recursiveMultipleDaysAheadForecast(1, [], startIndex)).reshape(1,forecastHorizon)

        else: 
            forecast = self.model.predict(xTestLSTM[startIndex:startIndex+1])
    
        return forecast

class outputLogCallback(Callback):

    def on_epoch_end(self, epoch, logs=None):

        if epoch % 100 == 0:
        
            print("Epoch: " + str(epoch) + " Validation Loss: " + \
                str(np.round(logs["val_loss"],6)) + " Training Loss: " + \
                    str(np.round(logs["loss"],6)))


    


