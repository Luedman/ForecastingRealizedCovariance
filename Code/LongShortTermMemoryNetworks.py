# tensorboard --logdir /Users/lukas/Desktop/HSG/2-Master/4_Masterthesis/NewCode --host=127.0.0.1
import numpy as np

from tensorflow.keras.optimizers       import Adam
from tensorflow.keras.models           import Sequential, load_model
from tensorflow.keras.layers           import Dense, LSTM, Dropout, add
from tensorflow.compat.v2.keras.layers import Bidirectional
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.callbacks        import TensorBoard, Callback, EarlyStopping
from tensorflow.keras.regularizers     import l1
import dataUtils

from tensorflow import __version__ as tfVersion
print("TF Version: " + str(tfVersion))

class LSTMmodel:

    def __init__(self, forecastHorizon = 30, 
                                    lookBack        = 32, 
                                    architecture    = [32], 
                                    dropout         = 0.0,
                                    regularization  = 0.0,
                                    loadPath        = None,
                                    bidirectional   = False):

        np.random.seed(1)
        self.modelType          = "LSTM"
        self.forecastHorizon    = forecastHorizon
        self.lookBack           = lookBack
        self.architecture       = architecture
        self.dropout            = dropout
        self.regularization     = l1(regularization)
        self.bidirectional      = bidirectional

        if loadPath is not None:
            self.model = load_model(loadPath)
            self.forecastHorizon = self.model.layers[-1].output_shape[1]
            self.lookBack = self.model.layers[0].output_shape[1]

            for layer in self.model.layers:
                if hasattr(layer, 'rate'):
                    self.dropout = layer.rate

            print("Model Loaded")
            #print(self.model.summary())
 
    def createModel(self, gpuOptmized = False):

        # This function takes self.architecture and creates 
        # a tensorflow model out of it 

        inputDimension = 1
        modelStructure = Sequential()
        
        def LSTMCell(nodes, **kwargs):

            def cpuLSTM(nodes, **kwargs):
                return LSTM(nodes, activation               ='relu',
                                    dropout                 = self.dropout,
                                    kernel_initializer      ='random_uniform',
                                    bias_initializer        ='ones',
                                    recurrent_regularizer   = self.regularization,
                                    **kwargs)
            
            def gpuLSTM(nodes, **kwargs):
                return CuDNNLSTM(nodes, kernel_initializer     ='random_uniform',
                                        bias_initializer        ='ones',
                                        recurrent_regularizer   = self.regularization,
                                        **kwargs)

            if not self.bidirectional:
                if gpuOptmized: return gpuLSTM(nodes, **kwargs)
                else:           return cpuLSTM(nodes, **kwargs)
 
            if self.bidirectional: 
                if gpuOptmized: return Bidirectional(gpuLSTM(nodes, **kwargs))
                else:           return Bidirectional(cpuLSTM(nodes, **kwargs))


        modelStructure.add(LSTMCell(self.architecture[0],
                                            return_sequences = (len(self.architecture) is not 1), 
                                            input_shape = (self.lookBack, inputDimension)))
        
        if len(self.architecture) > 1:

            for layerNo in range(1,len(self.architecture) - 1):
                modelStructure.add(LSTMCell(self.architecture[layerNo],
                                                return_sequences = True))

            modelStructure.add(LSTMCell(self.architecture[-1],
                                            return_sequences = False))

        modelStructure.add(Dense(self.forecastHorizon, activation='sigmoid'))
        
        self.model = modelStructure

    def train(self, epochs, 
                    xTrainLSTM, 
                    yTrainLSTM, 
                    xTestLSTM, 
                    yTestLSTM, 
                    modelname    = "LSTM", 
                    learningRate = 0.0001, 
                    batchSize    = 256, 
                    verbose      = 0):

        self.model.compile(loss = 'mse', optimizer = Adam(lr = learningRate, 
                                                            clipvalue = 0.5, 
                                                            decay     = 1e-12))

        tensorboard = TensorBoard(log_dir='./Log' + modelname, 
                                histogram_freq  = 0, 
                                write_graph     = False, 
                                write_images    = True)
                
        history = self.model.fit(xTrainLSTM, yTrainLSTM, 
                            epochs      = epochs, 
                            batch_size  = batchSize, 
                            verbose     = verbose, 
                            callbacks   = [tensorboard,overfitCallback], 
                            validation_data = (xTestLSTM, yTestLSTM))

        print("Model trained")
        print(self.model.summary())
        
        self.model.save(modelname +"-Model.h5")
        print("Model Saved")

        return history

    def multiStepAheadForecast(self, data, forecastHorizon, startIndex, windowMode, windowSize = 60):
        
        # If output shape is equal to one, perform recursive forecasting, otherwise do 
        # direct sequence to sequence forecasting

        if windowMode.upper() == "ROLLING":
            assert windowSize > 0, "multiStepAheadForecast: windowSize is not defined"
            newxTrain = np.concatenate([data.xTrainLSTM, data.xTestLSTM[:startIndex]])
            newyTrain = np.concatenate([data.yTrainLSTM, data.yTestLSTM[:startIndex]])
            self.model.fit(newxTrain[-windowSize:], newyTrain[-windowSize:], epochs = 100, batch_size = self.lookBack,verbose = 0, callbacks = [overfitCallback2])
            #self.model.fit(newxTrain[-windowSize:], newyTrain[-windowSize:], epochs = 1, batch_size = self.lookBack,verbose = 2)
            self.model.save("32a-Model.h5")

        elif windowMode.upper() == "EXPANDING":
            newxTrain = np.concatenate([data.xTrainLSTM, data.xTestLSTM[:startIndex]])
            newyTrain = np.concatenate([data.yTrainLSTM, data.yTestLSTM[:startIndex]])
            self.model.fit(newxTrain[:], newyTrain[:], epochs = 10, batch_size = self.lookBack,verbose = 0, callbacks = [overfitCallback2])
            #self.model.fit(newxTrain[:], newyTrain[:], epochs = 1, batch_size = self.lookBack,verbose = 2)
            self.model.save("32a-Model.h5")
        else: pass

        assert forecastHorizon == self.forecastHorizon or self.forecastHorizon == 1, \
            "Model has diffrent forecast Horizon"

        def recursiveMultipleDaysAheadForecast(daysAhead, forecasts, startIndex):

            xVectorLSTM = np.append(data.xTrainLSTM[:,0],data.xTestLSTM[:startIndex,0])
            backlog = np.append(xVectorLSTM, forecasts)[-self.lookBack:].reshape(1,self.lookBack,1)

            assert backlog.shape[1] == self.lookBack

            testPredict = self.model.predict(backlog)

            forecasts.append(testPredict[0][0])

            return recursiveMultipleDaysAheadForecast(daysAhead + 1, forecasts, startIndex)\
                 if (daysAhead < forecastHorizon) else forecasts

        if self.forecastHorizon == 1:
            forecast = np.array(recursiveMultipleDaysAheadForecast(1, [], startIndex)).reshape(forecastHorizon,1)
        else: 
            forecast = self.model.predict(data.xTestLSTM[startIndex:startIndex+1])
    
        return forecast

class outputLogCallback(Callback):

    def on_epoch_end(self, epoch, logs=None):

        if epoch % 100 == 0:
        
            print("Epoch: " + str(epoch) + " Validation Loss: " + \
                str(np.round(logs["val_loss"],6)) + " Training Loss: " + \
                    str(np.round(logs["loss"],6)))
# Log to Terminal
outputLog = outputLogCallback()

# Early Stopping
overfitCallback = EarlyStopping(monitor = 'val_loss', 
                                min_delta = 0.0001, 
                                patience = 50)

# Early Stopping
overfitCallback2 = EarlyStopping(monitor = 'loss', 
                                min_delta = 0.0001, 
                                patience = 5)


    


