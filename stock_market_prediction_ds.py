# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 11:03:02 2024

@author: Raghavendhra Devineni
"""

'''
Installing packages

Install the ucimlrepo package
    pip install ucimlrepo
    pytohn version: 3.9.13

'''


# !pip install ucimlrepo

from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# different numerical results due to floating-point round-off errors from different computation orders.
# To turn them off, set
TF_ENABLE_ONEDNN_OPTS=0

# # fetch dataset
# istanbul_stock_exchange = fetch_ucirepo(id=247)
ise_data = fetch_ucirepo(id=247)


# check the type of dataset
print("dataset type: ", type(ise_data), "\n")

# print the keys in dataset
print("keys: ", ise_data.keys(), "\n")

# fetch the stock data from the dataset
data_info = ise_data.data['features']
print("data type: ", type(data_info), "\n")

print("print first 5 rows..!")
print(data_info.head(), "\n")

#taking the reference variable
stock_data = data_info

# printing the summary statistics
# Descriptive statistics include those that summarize the central tendency, dispersion, and shape of a dataset’s distribution, excluding NaN values
print(stock_data.describe(),"\n \n")


# information about the dataset
print(stock_data.info(),"\n \n")

# print("\n","Data Columns: ", data_info.columns)

# print("varibles data :")
print(ise_data.variables,"\n \n")


# convert the date datatpe to datetime
print(stock_data['date'].head(),"\n \n")

stock_data['date'] = pd.to_datetime(stock_data['date'], format='%d-%b-%y')
print(stock_data['date'].head(),"\n", "\n")
# print(stock_data.dtypes, "/n")

# set the date column as Index
stock_data.set_index('date', inplace=True)

# #print few rows
# print("First few rows from the stock dataset...!")
# print(stock_data.head(),"\n \n")

# check for null values in the dataset
print("Check null values...!")
data_null_values = stock_data.isnull().sum()
print("null values found in dataset: ", data_null_values,"\n \n")

# calculating the mean of each month and restructure the data
print("Calculating the mean for each month...!")
stock_mean_data = stock_data.resample('M').mean()
print(stock_mean_data.head(),"\n \n")



# removing the duplicates in the columns
print("Removinig the duplicate columns...!")
stock_mean_data = stock_mean_data.loc[:, ~stock_mean_data.columns.duplicated()]
print(stock_mean_data.head(),"\n \n")

# Plot each column in the data
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10), constrained_layout=True)
data_columns = stock_mean_data.columns
for col, ax in enumerate(axes.flat):
    if col < len(data_columns):
        stock_mean_data[data_columns[col]].plot(ax=ax)
        ax.set_title(data_columns[col])
        ax.set_xlabel('year')
        ax.set_ylabel('price in mean')

plt.show()

# removing the duplicates in the columns from the dataset
print("Removinig the duplicate columns...!")
stock_data = stock_data.loc[:, ~stock_data.columns.duplicated()]
print(stock_data.head(),"\n \n")

# Identify the key features and targeted feature

X_stock_key_features = ['SP', 'DAX', 'FTSE', 'NIKKEI', 'BOVESPA', 'EU', 'EM'] # These features are used to train the model
Y_stock_target = ['ISE'] # to predict the price using the model


# SPlit the data into training and testing
def train_test_split_data(stock_data, X_stock_features, Y_stock_target):
  x_train, x_test, y_train, y_test = train_test_split(stock_data[X_stock_key_features], + stock_data[Y_stock_target],
                                                      test_size=0.2, random_state=42)

  return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = train_test_split_data(stock_data, X_stock_key_features, Y_stock_target)

print(len(x_train))

print(y_train.shape)
print(y_test.shape)

def normalize_data(x_train, x_test, y_train, y_test):

  # # normalize the data
  # # converting the stock prices to binary values into 0,1

  scaler_data =MinMaxScaler()

  norm_x_train = scaler_data.fit_transform(x_train) # features variable
  norm_x_test = scaler_data.transform(x_test)
  norm_y_train = scaler_data.fit_transform(y_train) # targeted variable
  norm_y_test = scaler_data.transform(y_test)

  return norm_x_train, norm_x_test, norm_y_train, norm_y_test

norm_x_train, norm_x_test, norm_y_train, norm_y_test = normalize_data(x_train, x_test, y_train, y_test)

print(norm_x_train.shape)
print(norm_y_train.shape)
print(norm_x_test.shape)
print(norm_y_test.shape)

# Reshape the normalized data by adding a timestep dimension
norm_x_train = norm_x_train.reshape((norm_x_train.shape[0], 1, norm_x_train.shape[1]))
norm_x_test = norm_x_test.reshape((norm_x_test.shape[0], 1, norm_x_test.shape[1]))

print(norm_x_train.shape)
print(norm_y_train.shape)
print(norm_x_test.shape)
print(norm_y_test.shape)

def lstm_model(norm_x_train):
  # Build the LSTM model from scratch
  model = Sequential()
  model.add(LSTM(units=128, return_sequences=True, input_shape=(norm_x_train.shape[1], norm_x_train.shape[2]))) #layer 1 with 50 units
  model.add(Dropout(0.3)) # prevent overfitting given 10% loss for every epoch
  model.add(LSTM(units=64, return_sequences=True)) # layer 2 with 64 units will only return the last output seq
  model.add(Dropout(0.3))
  model.add(LSTM(units=32))
  model.add(Dense(1)) # predicting only single values index(ISE) (predict single continuous value)


  model.compile(loss='mean_squared_error', #calculate the error (pred & act)
                optimizer=Adam(learning_rate=0.0003), #minimize the loss function.
                metrics=['mean_absolute_error']) # compiling the model


  return model




lstm_model = lstm_model(norm_x_train)



lstm_model.summary() # model summary

# train the model
history = lstm_model.fit(norm_x_train, norm_y_train, epochs=100, batch_size=16,
                         validation_split=0.2, verbose=1)


# predict the test data with lstm model
y_pred = lstm_model.predict(norm_x_test)
y_pred.shape

# just for checking
# predicting the training data
y_pred_train = lstm_model.predict(norm_x_train)
y_pred_train.shape


#Access the training data  to check the loss and MAE
training_hist = history.history

# evaluate model
train_loss, train_mean_absolute_error = lstm_model.evaluate(norm_x_test, norm_y_test)
print("\n")
print("training loss: ", train_loss)



# visualize the training loss and validation loss from training hstory
training_loss = training_hist['loss']
training_val_loss = training_hist['val_loss']
training_epochs = range(1, len(training_loss)+1)
plt.plot(training_epochs, training_loss, 'b', label='training loss')
plt.plot(training_epochs, training_val_loss, 'g', label='validation loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# visualize the training data, actual data and predicted data
plt.figure(figsize=(10,6))

# plot the training data
plt.plot(range(len(norm_y_train)), norm_y_train, label='Training data', color='blue')

# plot the actual data
plt.plot(range(len(norm_y_train), 536), norm_y_test, label='Original data', color='red')

# plot the predicted data
plt.plot(range(len(norm_y_train), 536), y_pred, label='Training data', color='black')

# Add the title and labels for the graphs
plt.title("Training, Actual vs Predicted")
plt.xlabel('')
plt.ylabel('')
plt.legend()
plt.show()

def calculate_error_rate(norm_y_test, y_pred):
  # check and convert predicted and norm_y_test are numpy(np) arrays
  y_pred = np.array(y_pred)
  norm_y_test = np.array(norm_y_test)

  # calculate the Mean Absolute Error (MAE)
  lstm_mae = mean_absolute_error(norm_y_test, y_pred)

  # calculate the Mean Absolute Percentage Error (MAPE)
  lstm_mape = mean_absolute_percentage_error(norm_y_test, y_pred)

  # calculate the Mean Squared Error (MSE)
  lstm_mse = mean_squared_error(norm_y_test, y_pred)

  # calculate the Root Mean Squared Error (RMSE)
  lstm_rmse = np.sqrt(lstm_mse)

  return lstm_mae, lstm_mape, lstm_mse, lstm_rmse


lstm_mae, lstm_mape, lstm_mse, lstm_rmse = calculate_error_rate(norm_y_test, y_pred)

# print the mena values
print(f"Mean absolute error: {lstm_mae} ")
print(f"Mean absolute percentage Error: {lstm_mape} ")
print(f"Mean squared Error: {lstm_mse} ")
print(f"Root mean square error: {lstm_rmse} ")



###################################################################
# Feature Extraction
###################################################################

import seaborn as sns


stock_data.head()

stock_corr_matrix_data = stock_data.corr()
stock_corr_matrix_data

# corr_matrix = stock_data.corr()
# corr_matrix

# Visualize the correlation matrix
plt.figure(figsize=(9,8))
sns.heatmap(stock_corr_matrix_data, annot=True, fmt=".2f")
plt.show()

# Extracted the unwanted features from the matrix
stock_feature_extracted_data = stock_data.drop(columns = ['FTSE','EU','EM'])
print(stock_feature_extracted_data.head())

# Get the correalation for the featured data
fe_stock_corr_matrix_data = stock_feature_extracted_data.corr()
print("\n",fe_stock_corr_matrix_data.head())

# Visualize the correlation matrix
plt.figure(figsize=(9,8))
sns.heatmap(fe_stock_corr_matrix_data, annot=True, fmt=".2f")
plt.show()

X_stock_key_features = ['SP', 'DAX', 'NIKKEI', 'BOVESPA']
Y_stock_target = ['ISE']

# split the extracted feature data into training and testing
x_train, x_test, y_train, y_test = train_test_split_data(stock_feature_extracted_data, X_stock_key_features, Y_stock_target)

print(len(x_train))

print(y_train.shape)
print(y_test.shape)

# normalize the featured stock data
fe_norm_x_train, fe_norm_x_test, fe_norm_y_train, fe_norm_y_test = normalize_data(x_train, x_test, y_train, y_test)

print(fe_norm_x_train.shape)
print(fe_norm_y_train.shape)
print(fe_norm_x_test.shape)
print(fe_norm_y_test.shape)

# Reshape the normalized data by adding a timestep dimension
fe_norm_x_train = fe_norm_x_train.reshape((fe_norm_x_train.shape[0], 1, fe_norm_x_train.shape[1]))
fe_norm_x_test = fe_norm_x_test.reshape((fe_norm_x_test.shape[0], 1, fe_norm_x_test.shape[1]))

print(fe_norm_x_train.shape)
print(fe_norm_y_train.shape)
print(fe_norm_x_test.shape)
print(fe_norm_y_test.shape)

def lstm_model(norm_x_train):
  # Build the LSTM model from scratch
  model = Sequential()
  model.add(LSTM(units=128, return_sequences=True, input_shape=(norm_x_train.shape[1], norm_x_train.shape[2]))) #layer 1 with 50 units
  model.add(Dropout(0.3)) # prevent overfitting given 10% loss for every epoch
  model.add(LSTM(units=64, return_sequences=True)) # layer 2 with 64 units will only return the last output seq
  model.add(Dropout(0.3))
  model.add(LSTM(units=32))
  model.add(Dense(1)) # predicting only single values index(ISE) (predict single continuous value)


  model.compile(loss='mean_squared_error', #calculate the error (pred & act)
                optimizer=Adam(learning_rate=0.0003), #minimize the loss function.
                metrics=['mean_absolute_error']) # compiling the model


  return model


fe_lstm_model = lstm_model(fe_norm_x_train)

# get the model summary
fe_lstm_model.summary()



# train the model
history1 = fe_lstm_model.fit(fe_norm_x_train, fe_norm_y_train, epochs=100, batch_size=16,
                         validation_split=0.2, verbose=1)

#Access the training data  to check the loss and MAE
training_hist = history1.history

# evaluate model
train_loss, train_mean_absolute_error = fe_lstm_model.evaluate(fe_norm_x_test, fe_norm_y_test)
print("\n")
print("training loss: ", train_loss)



# visualize the training loss and validation loss from training hstory
training_loss = training_hist['loss']
training_val_loss = training_hist['val_loss']
training_epochs = range(1, len(training_loss)+1)
plt.plot(training_epochs, training_loss, 'b', label='training loss')
plt.plot(training_epochs, training_val_loss, 'g', label='validation loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# predict the test data with lstm model
y_pred = fe_lstm_model.predict(fe_norm_x_test)
y_pred.shape

# visualize the training data, actual data and predicted data
plt.figure(figsize=(10,6))

# plot the training data
plt.plot(range(len(fe_norm_y_train)), fe_norm_y_train, label='Training data', color='#7D1007')

# plot the actual data
plt.plot(range(len(fe_norm_y_train), 536), fe_norm_y_test, label='Original data', color='#06C')

# plot the predicted data
plt.plot(range(len(fe_norm_y_train), 536), y_pred, label='Training data', color='#F6D173')

# Add the title and labels for the graphs
plt.title("Training, Actual vs Predicted")
plt.xlabel('')
plt.ylabel('')
plt.legend()
plt.show()



## Implementing ARIMA model

# !pip install pmdarima

from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX


# normalize the featured stock data
norm_x_train, norm_x_test, norm_y_train, norm_y_test = normalize_data(x_train, x_test, y_train, y_test)

print(norm_x_train.shape)
print(norm_y_train.shape)
print(norm_x_test.shape)
print(norm_y_test.shape)

# convert the norm_y_train and norm_y_test to 1D array through flatten
norm_y_train_flat = norm_y_train.flatten()
norm_y_test_flat = norm_y_test.flatten()

print(norm_y_train_flat.shape)
print(norm_y_test_flat.shape)

# Build ARIMA model,
# Find the best order for AR(P), I(q), M(d) parameters using auto_arima,
# set the seasonality = True, to get the best sasonality order.

arima_auto_model = auto_arima(norm_y_train_flat, exogenous=norm_x_train, seasonal=False,
                              trace=True, error_action='ignore',
                              suppress_warnings=True, stepwise=True)


# Identify the best order through auto_arima
arima_best_order = arima_auto_model.order
arima_best_order

# Use the SARIMAX model using the optimal order to train 
# SARIMAX use both features and target for training the data unlike ARIMA it accepts only target data
arima_model = SARIMAX(norm_y_train_flat, exog=norm_x_train, order=arima_best_order)
arima_model = arima_model.fit(disp=False)

# summarize the arima_model
arima_model.summary()

# use the model to predict on the training data
armia_predict = arima_model.predict(start=0, end=len(norm_y_train_flat)-1,
                                    exog=norm_x_train)
armia_predict.shape

# Forecast the model on the test data (target)
arima_forecast = arima_model.forecast(steps=len(norm_y_test_flat), exog=norm_x_test)
arima_forecast.shape

# visualize the data
plt.figure(figsize=(10,6))
plt.plot(norm_y_train_flat, label='Training data', color='g')
plt.plot(armia_predict, label='Predicted data', color='red')
fore_start_index = len(norm_y_train_flat) + np.arange(len(arima_forecast))
plt.plot(fore_start_index, arima_forecast, label='Forecast data', color='black')
plt.title('')
plt.xlabel('')
plt.ylabel('')
plt.legend()
plt.show()

lstm_mae, lstm_mape, lstm_mse, lstm_rmse = calculate_error_rate(norm_y_test, arima_forecast)

# print the mena values
print(f"Mean absolute error: {lstm_mae} ")
print(f"Mean absolute percentage Error: {lstm_mape} ")
print(f"Mean squared Error: {lstm_mse} ")
print(f"Root mean square error: {lstm_rmse} ")