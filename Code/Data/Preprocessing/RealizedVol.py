
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
import datetime as dat
import xlsxwriter

from keras.optimizers       import Adam
from keras.models           import Sequential
from keras.layers           import Dense
from keras.layers           import LSTM


# Hyperparameters
look_back = 10
dropout = 0.2
epochs = 100


usecols = ['Dates', 
           'DJI_rv', 
           'FCHI_rv', 
           'FTSE_rv', 
           'IBEX_rv']

data = pd.read_excel('realizedlibrary01.xls', skiprows = 1, usecols = usecols)
data['Dates'] = pd.to_datetime(data['Dates'], format='%Y%m%d')
data = data.set_index('Dates')

# Find a better solution for gaps in the data
data = data.interpolate(limit = 2) 
data = data.fillna(0)

# Normalize the values
scaler = MinMaxScaler(feature_range=(0, 1))

for cols in usecols[1:]:
    data['Norm'+ cols] = scaler.fit_transform(data[cols].values.reshape(len(data[cols]),1))


def create_dataset(input_series):
    
    # Creates a training and test set that consist of time series and a 1 day ahead label
    # input series is an numpy array of shape (?,) or (?,1)
    
    input_series = input_series.reshape(len(input_series),1)

    def data_split(input_series, split = 0.80):

        # Split the data into a training and test set
        # Input series is a numpy array
        # We are using 80 percent (default) of the data as training set and 20% as the test set

        train_size  = int(len(input_series) * split)
        test_size   = len(input_series) - train_size

        train, test = input_series[0:train_size], input_series[train_size:len(input_series)]

        return train, test

    def create_timeseries(time_series, look_back):

        # dataX is the is the rolling window of past oberservations 
        # dataY becomes the the value that is one day ahead of the rolling window. 
        # This is the label/prediction for the past values

        dataX, dataY = [], []

        for i in range(1,len(time_series) - look_back - 1):

            x = time_series[i:i + look_back]
            dataX.append(x)

            y = time_series[i + look_back + 1]
            dataY.append(y)

        return np.array(dataX), np.array(dataY)

    # Create the dataset with rolling window for the training set and test set
    trainX, trainY  = create_timeseries(data_split(input_series)[0], look_back)
    testX, testY    = create_timeseries(data_split(input_series)[1], look_back)

    # Reshape input to be [samples, time steps, features]
    trainX  = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX   = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    
    return trainX, trainY, testX, testY

def define_model(nodes):
    
    hidden_layers = len(nodes) - 1 
    
    # Define Input layer
    model = Sequential()
    model.add(LSTM(nodes[0], return_sequences = True, input_shape = (1, look_back)))
    
    # Add hidden Layers
    for i in range(hidden_layers - 1):
        model.add(LSTM(nodes[i+1], return_sequences = True, dropout = dropout))
    
    # Define last hidden layer
    model.add(LSTM(nodes[-1]))
    
    # Define output layer
    model.add(Dense(1,  activation = "linear"))

    # Compile Model
    model.compile(loss = 'mean_squared_error', optimizer = Adam(lr = 0.0001))
              
    #print(model.summary())
    
    return model

def evaluate_model(input_series, nodes):
    
    trainX, trainY, testX, testY = create_dataset(input_series)
    
    def train_model(input_series, nodes): 
    
        model = define_model(nodes)

        model.fit(trainX, trainY, epochs = epochs, batch_size = 1, verbose = 0)

        #model.save('Output/Model_LB-'+ str(look_back) + '_EP-'+str(epochs)+'.h5')

        return model

    def calculate_error(model, trainX, trainY, testX, testY):
    
        # Make predictions 
        trainPredict    = model.predict(trainX)
        testPredict     = model.predict(testX)

        # Inverse the normalization procedure of the data
        trainY = np.reshape(trainY,(trainY.shape[0],))
        testY  = np.reshape(testY,(testY.shape[0],))

        trainPredict    = scaler.inverse_transform(trainPredict)
        trainY          = scaler.inverse_transform([trainY])
        testPredict     = scaler.inverse_transform(testPredict)
        testY           = scaler.inverse_transform([testY])

        # Calculate root mean squared error
        trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
        testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
        
        
        return trainScore, testScore
    
    
    model = train_model(input_series, nodes)
        
    trainScore, testScore = calculate_error(model, trainX, trainY, testX, testY)
    
    return trainScore, testScore


def create_architectures(max_nodes, max_layers):

    architectures = []
    
    for layers in range(1,max_layers + 1):
        
        for nodes in range(5, max_nodes + 5, 5):
    
            node_structure = []

            for i in range(layers):

                    node_structure.append(int(np.ceil(nodes - nodes/max_layers*i)))

            architectures.append(node_structure)

    return architectures



def initialize_training_series(data, max_nodes, max_layers):
    
    architectures = create_architectures(max_nodes,max_layers)
    
    results = []
    i = 0

    excel_log = xlsxwriter.Workbook('./ExcelLogs/Training_Results_' + dat.datetime.now().strftime("%Y-%m-%d--%H-%M-%S" ) + '.xlsx')
    worksheet = excel_log.add_worksheet()
    
    for architecture in architectures:
        
        trainScore, testScore = evaluate_model(data, architecture)
        
        dt = dat.datetime.now().strftime("%Y-%m-%d--%H-%M-%S" )

        print(dt + '  ' + str(architecture))
        print('Train Score: %.6f RMSE' % (trainScore))
        print('Test Score: %.6f RMSE' % (testScore))
        
        results.append([str(architecture), trainScore, testScore])

        worksheet.write(i, 1, str(architecture))
        worksheet.write(i, 2, trainScore)
        worksheet.write(i, 3, testScore)
        
        i += 1

    excel_log.close()
    print(max(results, key = lambda x: x[2]))


initialize_training_series(data["NormDJI_rv"].values, 30, 3)
