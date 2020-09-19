#!/usr/bin/python3
# Reference: https://medium.com/@randerson112358/stock-price-prediction-using-python-machine-learning-e82a039ac2bb
import math
import pandas_datareader.data as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint, ModelCheckpoint
import time, sys

days_train_period = 30  # 30, 60, 90, 180
tf.random.set_seed(37282760123562386)
plt.style.use('fivethirtyeight')

if len(sys.argv) != 2:
    print("\nError, eg. use: stock-price-pred AAPL")
    exit(1)

dir = '/home/developer/Develop/profesional/stock-price/'
stock_name = sys.argv[1]
start = '01/01/2010'
# Get the stock quote (AAPL, FB, ZM, TSLA, AMD, AMZN)
df = web.DataReader(stock_name, data_source='yahoo', start=start)
# Use Close attribute as dataset
data = df.filter(['Close'])
dataset = data.values

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

training_data_len = math.ceil(len(scaled_data) * 0.8)
predictions = []


def create_model(train_data):
    # Split the data into x_train and y_train data sets
    x_train = []
    y_train = []

    for i in range(days_train_period, len(train_data)):
        x_train.append(train_data[i-days_train_period:i, 0])
        y_train.append(train_data[i, 0])

    # Convert data sets in numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    # print(x_train.shape, y_train.shape)

    ##############################################################
    ##############################################################
    ##############################################################

    # Build the LSTM model
    # model = Sequential()
    # model.add(LSTM(60, input_shape=(x_train.shape[1], 1), return_sequences=True))
    # model.add(Dropout(0.2))
    # model.add(BatchNormalization())

    # model.add(LSTM(60, return_sequences=True))
    # model.add(Dropout(0.1))
    # model.add(BatchNormalization())

    # model.add(LSTM(60))
    # model.add(Dropout(0.2))
    # model.add(BatchNormalization())

    # model.add(Dense(30, activation='relu'))
    # model.add(Dropout(0.2))

    # model.add(Dense(1))

    # opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
    # # Compile model
    # model.compile(
    #     loss='mean_squared_error',
    #     optimizer=opt,
    #     metrics=['accuracy']
    # )

    # # RATIO_TO_PREDICT = "LTC-USD"
    # EPOCHS = 1  # how many passes through our data
    # BATCH_SIZE = 1  # how many batches? Try smaller batch if y

    # # Train model
    # history = model.fit(
    #     x_train, y_train,
    #     batch_size=BATCH_SIZE,
    #     epochs=EPOCHS,
    # )

    model = Sequential()
    model.add(LSTM(units=int(days_train_period*2),
                   return_sequences=True, input_shape=(x_train.shape[1], 1)))
    # model.add(LSTM(units=int(days_train_period*2),
    #                return_sequences=True))
    model.add(LSTM(units=int(days_train_period*2), return_sequences=False))
    model.add(Dense(units=int(days_train_period)))
    model.add(Dense(units=1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error',
                  metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, batch_size=1, epochs=3)

    return model


def export_model(model, filename_model):
    # Save the model
    model.save(filename_model)


def import_model(filename_model):
    # Recrea exactamente el mismo modelo solo desde el archivo
    return load_model(filename_model)


# model = create_model(scaled_data[:training_data_len])
# export_model(model, 'dir + StockPricePrediction.model.h5')

model = import_model(dir + 'StockPricePrediction.model.h5')

#################################


def test(test_data):
    global predictions
    x_test = []

    for i in range(days_train_period, len(test_data)):
        # Create the x_test and y_test data sets
        x_test.append(test_data[i-days_train_period:i, 0])

    # Get all of the rows from index 1603 to the rest and all of the columns (in this case it's only column 'Close'), so 2003 - 1603 = 400 rows of data
    y_test = scaled_data[training_data_len:, :]

    # Convert x_test to a numpy array
    x_test = np.array(x_test)

    # Reshape the data into the shape accepted by the LSTM
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # print(x_test)
    # Getting the models predicted price values
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)  # Undo scaling

    # Calculate/Get the value of RMSE
    rmse = np.sqrt(np.mean(((predictions - y_test)**2)))
    print('rmse:', rmse, '\n')


# Test data set
test_data = scaled_data[training_data_len - days_train_period:, :]
test(test_data)

##############################################################
##############################################################
##############################################################


def show_graph():
    # Plot/Create the data for the graph
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions
    # Visualize the data
    plt.figure(figsize=(16, 8))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.show()


def show_table():
    # Show the valid and predicted prices
    valid = data[training_data_len:]
    valid['Predictions'] = predictions
    print(valid)

##########################################################


def get_quote(model, name):
    # Get the quote
    apple_quote = web.DataReader(name, data_source='yahoo', start=start)
    # Create a new dataframe
    new_df = apple_quote.filter(['Close'])
    # Get teh last day closing price
    last_days = new_df[-days_train_period:].values
    # Scale the data to be values between 0 and 1
    last_days_scaled = scaler.transform(last_days)

    X_test = []
    X_test.append(last_days_scaled)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Get the predicted scaled price
    pred_price = model.predict(X_test)
    # undo the scaling
    pred_price = scaler.inverse_transform(pred_price)

    return pred_price


show_table()

pred_price = get_quote(model, stock_name)
print('pred price for ' + stock_name + ':', pred_price)

show_graph()