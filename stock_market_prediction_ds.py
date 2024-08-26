# -*- coding: utf-8 -*-
"""
Created on Mon Ju1 5 22:24:49 2024

@author: Raghavendhra Devineni

Original file is located at
    https://colab.research.google.com/drive/13czNhF8Tq52Yguwa3KBaxwFkGODUmqqG
"""


# Install the ucimlrepo to download the dataset from UCL repository
# !pip install ucimlrepo

# Import necessary libraries for data preprocessing and model training
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
import seaborn as sns

# Different numerical results due to floating-point round-off errors from different computation orders.
# To turn them off, set TF_ENABLE_ONEDNN_OPTS = 0
TF_ENABLE_ONEDNN_OPTS=0

# Fetch dataset istanbul stock exchange dataset id
ise_data = fetch_ucirepo(id=247)

# Check the type of dataset
print("dataset type: ", type(ise_data), "\n")

# Print the keys in dataset
print("keys: ", ise_data.keys(), "\n")

# Fetch the stock data from the dataset
data_info = ise_data.data['features']
print("data type: ", type(data_info), "\n")

# print first five rows of the Dataframe to inspect
print("print first 5 rows..!")
print(data_info.head(), "\n")

# Taking new reference variable for testing
stock_data = data_info

# printing the summary statistics

# print the descriptive statistics includes that summarize the tendency, dispersion & shape of the dataset, exclusing NaN values
print(stock_data.describe(),"\n \n")

# print information about the dataset, including the data types
print(stock_data.info(),"\n \n")

# print the variable/columns of the dataset
print(ise_data.variables,"\n \n")

# print first few rows of date column before converting to datetime format
print(stock_data['date'].head(),"\n \n")

# convert the data to Datetime type
stock_data['date'] = pd.to_datetime(stock_data['date'], format='%d-%b-%y')

# print first few rows of the data
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
print("Removing the duplicate columns...!")
stock_mean_data = stock_mean_data.loc[:, ~stock_mean_data.columns.duplicated()]

# print first few rows of the data
print(stock_mean_data.head(),"\n \n")

# Plot each column in the data

# Create a 3x3 grid of subplots with a specified figure size and layout constraints
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10), constrained_layout=True)
data_columns = stock_mean_data.columns
for col, ax in enumerate(axes.flat):
    if col < len(data_columns):
        # Plot the mean data for each column
        stock_mean_data[data_columns[col]].plot(ax=ax)
        # Set the title and labels for each subplot
        ax.set_title(data_columns[col])
        ax.set_xlabel('year')
        ax.set_ylabel('price in mean')

# show the plots
plt.show()

# removing the duplicates in the columns from the dataset
print("Removing the duplicate columns...!")
stock_data = stock_data.loc[:, ~stock_data.columns.duplicated()]
print(stock_data.head(),"\n \n")

# Identify the key features and targeted feature
# These features are used to train the model
X_stock_key_features = ['SP', 'DAX', 'FTSE', 'NIKKEI', 'BOVESPA', 'EU', 'EM']
# Target featuer used for forecasting the future
Y_stock_target = ['ISE']

def train_test_split_data(stock_data, X_stock_features, Y_stock_target):
  '''
  Split the stock data into training and testing

  Parameters:
  stock_data(Dataframe): The Dataframe containing the stock data
  X_stock_features : List of the feature columns (X) to be used as input
  Y_stock_target : Column name of the target variable (Y) to be predicted.

  Returns:
  x_train, x_test, y_train, y_test : Returns the training and testing for both features

  '''
  # split the data into training and testing with testsize 20%
  # random_state=42 ensure that split data is reproducible
  x_train, x_test, y_train, y_test = train_test_split(stock_data[X_stock_key_features], + stock_data[Y_stock_target],
                                                      test_size=0.2, random_state=42)

  return x_train, x_test, y_train, y_test

# Function used to split the training and testing sets
x_train, x_test, y_train, y_test = train_test_split_data(stock_data, X_stock_key_features, Y_stock_target)

print(len(x_train))

print(y_train.shape)
print(y_test.shape)

"""# **Normalization**"""

def normalize_data(x_train, x_test, y_train, y_test):
  """
  This function used to normalize the training and testing data.

  Parameters:
  x_train, x_test : These are the training and testing features
  y_train, y_test : These are the training and testing target variables

  Returns:
  norm_x_train, norm_x_test, norm_y_train, norm_y_test : Returns the normalized features and target.
  """
  # Initilize the scalar
  scaler_data =MinMaxScaler()

  norm_x_train = scaler_data.fit_transform(x_train) # features variable
  norm_x_test = scaler_data.transform(x_test)
  norm_y_train = scaler_data.fit_transform(y_train) # targeted variable
  norm_y_test = scaler_data.transform(y_test)

  return norm_x_train, norm_x_test, norm_y_train, norm_y_test

# Function used to normalize the features and target variables
norm_x_train, norm_x_test, norm_y_train, norm_y_test = normalize_data(x_train, x_test, y_train, y_test)

# print the shape of each variables
# Here X is the rows and y is the columns
print(norm_x_train.shape)
print(norm_y_train.shape)
print(norm_x_test.shape)
print(norm_y_test.shape)

# Reshape the normalized data by adding a timestep dimension
# here 1 is the timestep
norm_x_train = norm_x_train.reshape((norm_x_train.shape[0], 1, norm_x_train.shape[1]))
norm_x_test = norm_x_test.reshape((norm_x_test.shape[0], 1, norm_x_test.shape[1]))

print(norm_x_train.shape)
print(norm_y_train.shape)
print(norm_x_test.shape)
print(norm_y_test.shape)

"""# **Implementing LSTM MODEL**"""

def lstm_model(norm_x_train):
  '''
  Build the LSTM model from scratch for timeseries forecasting
  Parameters:
  norm_x_train: normalize the training data used to define the Input shape
  Returns:
  model: Returns the compiled LSTM model ready for training.
  '''

  # Initial the sequential model
  model = Sequential()
  # Layer 1 with 128 units
  # Return_sequences=True means this layer will return full sequence output
  model.add(LSTM(units=128, return_sequences=True, input_shape=(norm_x_train.shape[1], norm_x_train.shape[2])))
  # Adding dropout layer with a rate of 0.3 (30%) to prevent overfittinig
  model.add(Dropout(0.3))

  # Layer 2 with 64 units
  model.add(LSTM(units=64, return_sequences=True))
  # Adding dropout layer with a rate of 0.3 (30%) to prevent overfittinig
  model.add(Dropout(0.3))

  # Layer 3 with 32 units
  model.add(LSTM(units=32))
  # Output layer: Dense layer with a single neuron to predict the target value
  model.add(Dense(1)) # predicting only single values index(ISE) (predict single continuous value)

  # compile the model
  model.compile(loss='mean_squared_error', #calculate the error (pred & act)
                optimizer=Adam(learning_rate=0.0003), #minimize the loss function.
                metrics=['mean_absolute_error']) # compiling the model


  return model

# function calling the lstm model using the normalized training data
# to define the input shape
lstm_model = lstm_model(norm_x_train)

# model summary
lstm_model.summary()

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

#Access the training data  to check the training loss and Mean Absolute Error (MAE)
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
# Set the title and labels for plot
plt.title("LSTM Model Training-loss vs Validation-loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
# Add legend to differentiate between the lines on graph
plt.legend()
# show the plot
plt.show()

# visualize the training data, actual data and predicted data
plt.figure(figsize=(10,6))

# plot the training data, represent the normalized training data
plt.plot(range(len(norm_y_train)), norm_y_train, label='Training data', color='blue')

# plot the actual data, represent the normalized y_test data
plt.plot(range(len(norm_y_train), 536), norm_y_test, label='Original data', color='red')

# plot the predicted data, represent the prediction made by the model
plt.plot(range(len(norm_y_train), 536), y_pred, label='Training data', color='black')

# Add the title and labels for the graphs
# legend to differentiate between the lines and graph
plt.title("Comparison of Training, Actual and Predicted")
plt.xlabel('Time Steps')
plt.ylabel('Normalized values')
plt.legend()
# show the plot
plt.show()

def calculate_error_rate(norm_y_test, y_pred):
  # check and convert the predicted and norm_y_test are numpy(np) arrays
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

# FUnction used to calculate the error metrics
lstm_mae, lstm_mape, lstm_mse, lstm_rmse = calculate_error_rate(norm_y_test, y_pred)

# print the error metrics values
print(f"Mean absolute error: {lstm_mae} ")
print(f"Mean absolute percentage Error: {lstm_mape} ")
print(f"Mean squared Error: {lstm_mse} ")
print(f"Root mean square error: {lstm_rmse} ")

"""# **LSTM model with feature extraction**"""

# print first few rows of the dataset
stock_data.head()

# Calculate the correlation matrix for the stock data
# This will show the pairwise correlation between all columns in the dataframe
stock_corr_matrix_data = stock_data.corr()

# Display the correlation matrics
stock_corr_matrix_data

# Visualize the correlation matrix using a heat map
# Set the size of the figure
plt.figure(figsize=(9,8))
# set annot=True, add the correlation values on the heatmap
# fmt=".2f" format the correlation coefficient to two decimal places
sns.heatmap(stock_corr_matrix_data, annot=True, fmt=".2f")

# show the heat map
plt.show()

# Extracted the unwanted features from the matrix
stock_feature_extracted_data = stock_data.drop(columns = ['FTSE','EU','EM'])
print(stock_feature_extracted_data.head())

# Get the correalation for the featured data
fe_stock_corr_matrix_data = stock_feature_extracted_data.corr()
print("\n",fe_stock_corr_matrix_data.head())

# Visualize the correlation matrix using a heat map
# Set the size of the figure
plt.figure(figsize=(9,8))
# set annot=True, add the correlation values on the heatmap
# fmt=".2f" format the correlation coefficient to two decimal places
sns.heatmap(fe_stock_corr_matrix_data, annot=True, fmt=".2f")

# show the heat map
plt.show()

# Identify the key features and targeted feature
# These features are used to train the model
X_stock_key_features = ['SP', 'DAX', 'NIKKEI', 'BOVESPA']
# Target featuer used for forecasting the future
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


# function calling the lstm model using the normalized training data
# to define the input shape
fe_lstm_model = lstm_model(fe_norm_x_train)

# get the model summary for featured extraction
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
# Set the title and labels for plot
plt.title("LSTM Model Training-loss vs Validation-loss (After feature extraction)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
# Add legend to differentiate between the lines on graph
plt.legend()
# show the plot
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
plt.xlabel('Time Steps')
plt.ylabel('Normalized Values')
plt.legend()
plt.show()



# Calculate the error metrics
lstm_mae, lstm_mape, lstm_mse, lstm_rmse = calculate_error_rate(fe_norm_y_test, y_pred)

# print the error metrics values
print(f"Mean absolute error: {lstm_mae} ")
print(f"Mean absolute percentage Error: {lstm_mape} ")
print(f"Mean squared Error: {lstm_mse} ")
print(f"Root mean square error: {lstm_rmse} ")

"""## Implementing ARIMA model"""

# install the pmdarimaa library
# !pip install pmdarima

# Import the necessary libraries
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Identify the key features and targeted feature
# These features are used to train the model
X_stock_key_features = ['SP', 'DAX', 'FTSE', 'NIKKEI', 'BOVESPA', 'EU', 'EM']
# Target featuer used for forecasting the future
Y_stock_target = ['ISE']

# split the extracted feature data into training and testing
x_train, x_test, y_train, y_test = train_test_split_data(stock_data, X_stock_key_features, Y_stock_target)

print(len(x_train))

print(y_train.shape)
print(y_test.shape)

# Function used to normalize the features and target variables
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
# print the shape of the model
armia_predict.shape

# Forecast the model on the test data (target)
arima_forecast = arima_model.forecast(steps=len(norm_y_test_flat), exog=norm_x_test)
# print the shape of the model
arima_forecast.shape

# visualize the data
# Set the figure size
plt.figure(figsize=(10,6))
# plot the training data and predicted data
plt.plot(norm_y_train_flat, label='Training data', color='g')
plt.plot(armia_predict, label='Predicted data', color='red')

# Add the title and labels for the graphs
# legend to differentiate between the lines and graph
plt.title('Comparision of Training vs Predicted data')
plt.xlabel('Time')
plt.ylabel('Normalized Values')
plt.legend()
# show the plot
plt.show()

# visualize the data
# Set the figure size
plt.figure(figsize=(10,6))
# plot the training data, predicted data, and forecast data
plt.plot(norm_y_train_flat, label='Training data', color='g')
plt.plot(armia_predict, label='Predicted data', color='red')
fore_start_index = len(norm_y_train_flat) + np.arange(len(arima_forecast))
plt.plot(fore_start_index, arima_forecast, label='Forecast data', color='black')
# Add the title and labels for the graphs
# legend to differentiate between the lines and graph
plt.title('Comparision of Training, Predicted and Forecast data')
plt.xlabel('Time')
plt.ylabel('Normalized Vlaues')
plt.legend()
# show the plot
plt.show()

# Function used to calculate the error metrics
lstm_mae, lstm_mape, lstm_mse, lstm_rmse = calculate_error_rate(norm_y_test, arima_forecast)

# print the error metricsvalues
print(f"Mean absolute error: {lstm_mae} ")
print(f"Mean absolute percentage Error: {lstm_mape} ")
print(f"Mean squared Error: {lstm_mse} ")
print(f"Root mean square error: {lstm_rmse} ")

"""# **ARIMA Feature extraction**"""

# Identify the key features and targeted feature
# These features are used to train the model
X_stock_key_features = ['SP', 'DAX', 'NIKKEI', 'BOVESPA']
# Target feature used for forecasting the future
Y_stock_target = ['ISE']

# split the extracted feature data into training and testing
x_train, x_test, y_train, y_test = train_test_split_data(stock_feature_extracted_data, X_stock_key_features, Y_stock_target)

print(len(x_train))

print(y_train.shape)
print(y_test.shape)

# Function used to normalize the features and target variables
norm_x_train, norm_x_test, norm_y_train, norm_y_test = normalize_data(x_train, x_test, y_train, y_test)

print(norm_x_train.shape)
print(norm_y_train.shape)
print(norm_x_test.shape)
print(norm_y_test.shape)

# convert the norm_y_train and norm_y_test to 1D array through flatten
norm_y_train_flat = norm_y_train.flatten()
norm_y_test_flat = norm_y_test.flatten()

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
# set the figure size
plt.figure(figsize=(10,6))
# plot the training data, predicted and Forecast data
plt.plot(norm_y_train_flat, label='Training data', color='g')
plt.plot(armia_predict, label='Predicted data', color='red')
fore_start_index = len(norm_y_train_flat) + np.arange(len(arima_forecast))
plt.plot(fore_start_index, arima_forecast, label='Forecast data', color='black')
# Add the title and labels for the graphs
# legend to differentiate between the lines and graph
plt.title('Comparision of Training, Predicted and Forecast data (After Feature Extraction)')
plt.xlabel('Time Step')
plt.ylabel('Normalized Vlaues')
plt.legend()
# show the plot
plt.show()

# Function used to calculate the error metrics
lstm_mae, lstm_mape, lstm_mse, lstm_rmse = calculate_error_rate(norm_y_test, arima_forecast)

# print theerror metrics values
print(f"Mean absolute error: {lstm_mae} ")
print(f"Mean absolute percentage Error: {lstm_mape} ")
print(f"Mean squared Error: {lstm_mse} ")
print(f"Root mean square error: {lstm_rmse} ")

"""# **Implementing Prophet model**"""

# Install the prophet library created by Facebook
# !pip install prophet

# Import the necessary library
from prophet import Prophet

# reset the index to the  stock dataset
fbp_data = stock_data.reset_index()
fbp_data.head()

# check the date column is in datetime format
# rename the columns as date as 'ds' and ISE as 'y' (target column)
fbp_data['ds'] = pd.to_datetime(fbp_data['date'])
fbp_data = fbp_data.rename(columns={'ISE':'y'})
fbp_data.head()

# split the data into training and testing for prophet model
# assign the 80% for testing and 20% for testing
training_size = int(len(fbp_data)*0.8)
fbp_train = fbp_data[:training_size]
fbp_test = fbp_data[training_size:]
print(fbp_train.shape)
print(fbp_test.shape)

# create prophet model
prophet_model = Prophet(
    seasonality_mode='multiplicative',  # Ensure that multiplicative seasonality is applied
    changepoint_prior_scale=2.0,  # Maximum flexibility to capture trend changes
    seasonality_prior_scale=20.0  # Allow more variability in seasonality
)

# fit the model
prophet_model.fit(fbp_train)

# get the future values in the dataframe
fbp_model_future = prophet_model.make_future_dataframe(periods=len(fbp_test), freq='D')
# Add the regressor columns to the model_future dataframe
fbp_model_future = pd.concat([fbp_model_future, fbp_data.drop(['ds', 'y'], axis=1)], axis=1).iloc[-len(fbp_model_future):]
# predict the vales
model_predict = prophet_model.predict(fbp_model_future)

print(fbp_test.shape)
print(fbp_model_future.shape)
print(model_predict.shape)

# check the forecast values for the test data set
forecast_test_data = model_predict[model_predict['ds'] >= fbp_test['ds'].min()]
forecast_test_data.shape

# visualize the data
plt.figure(figsize=(10,6))

# plot the actual data
plt.plot(fbp_data['ds'], fbp_data['y'], label='Original data', color='green', linewidth=2)

# plot the forecast data for test dataset
plt.plot(forecast_test_data['ds'], forecast_test_data['yhat'], label='Forecast data', color='blue', linewidth=2)

# Fill the area between the forecast bounds
plt.fill_between(forecast_test_data['ds'], forecast_test_data['yhat_lower'], forecast_test_data['yhat_upper'], color='gray', alpha=0.2)

# customize the plot
plt.legend()
plt.xlabel('Date')
plt.ylabel('ISE')
plt.title('Actual vs Forecast')

# show the plot
plt.show()

# Calculate the error matrics

# Merge the forecasted values with the actual data
predicted_data = fbp_test.copy()
predicted_data = predicted_data.merge(forecast_test_data[['ds', 'yhat']], on='ds')

# calculate the MAE
fbp_mae = mean_absolute_error(predicted_data['y'], predicted_data['yhat'])

# claculate the RMSE
fbp_rmse = np.sqrt(mean_squared_error(predicted_data['y'], predicted_data['yhat']))

print(f"Mean Absolute Error (MAE):{fbp_mae}")
print(f"Root Mean sqaure Error (RMSE):{fbp_rmse}")