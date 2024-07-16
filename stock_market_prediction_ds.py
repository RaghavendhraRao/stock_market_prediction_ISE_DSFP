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

from ucimlrepo import fetch_ucirepo 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout


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

stock_data = data_info
# print(stock_data.head())

# printing the summary statistics
# Descriptive statistics include those that summarize the central tendency, dispersion, and shape of a dataset’s distribution, excluding NaN values
print(stock_data.describe())


# information about the dataset
print(stock_data.info())

# print("\n","Data Columns: ", data_info.columns)

# print("varibles data :")
print(ise_data.variables,"\n")


# convert the date datatpe to datetime
print(stock_data['date'].head(),"\n")

stock_data['date'] = pd.to_datetime(stock_data['date'], format='%d-%b-%y')
print(stock_data['date'].head(),"\n", "\n")
print(stock_data.dtypes, "/n")

# set the date column as Index
stock_data.set_index('date', inplace=True)

#print few rows
print(stock_data.head())

# check for null values in the dataset
data_null_values = stock_data.isnull().sum()
print("null values found in dataset: ", data_null_values, "/n")

# calculating the mean of each month and restructure the data
stock_mean_data = stock_data.resample('M').mean()
print(stock_mean_data.head(), "/n")

# removing the duplicates in the columns
stock_mean_data = stock_mean_data.loc[:, ~stock_mean_data.columns.duplicated()]
print(stock_mean_data.head(), "/n")

# Plot each column
# fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10), constrained_layout=True)
# data_columns = stock_mean_data.columns
# for col, ax in enumerate(axes.flat):
#     if col < len(data_columns):
#         stock_mean_data[data_columns[col]].plot(ax=ax)
#         ax.set_title(data_columns[col])
#         ax.set_xlabel('year')
#         ax.set_ylabel('price in mean')
        
# plt.show()


# # split data into training and testing using Time-Based split(year, months, days)
# stock_train_data = stock_mean_data['2009': '2010-02']
# stock_test_data = stock_mean_data['2010-03':]

X_stock_key_features = ['SP', 'DAX', 'FTSE', 'NIKKEI', 'BOVESPA', 'EU', 'EM']
Y_stock_target = ['ISE']



# normalize the data
scaler_data =MinMaxScaler(feature_range=(0,1))
normalized_data = scaler_data.fit_transform(stock_mean_data[X_stock_key_features + Y_stock_target])

# split the data into features and target
X_data = normalized_data[:, :-1]
Y_data = normalized_data[:, -1]


#Reshape the input data into 3D array/shape for LSTM
allow_timestep =1
X_data = np.reshape(X_data, (X_data.shape[0], allow_timestep, X_data.shape[1]))


# split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X_data, Y_data,
                                                    test_size=0.7, random_state=42)

# Build the LSTM model from scratch
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2]))) #layer 1 with 50 units
model.add(Dropout(0.1)) # prevent overfitting given 10% loss for every epoch
model.add(LSTM(units=50)) # layer 2 with 50 units will only return the last output seq
model.add(Dense(1)) # predicting only single values index(ISE) (pridict single continuous value)
model.compile(loss='mean_squared_error', #calculate the error (pred & act)
              optimizer='adam', #minimize the loss function.
              metrics=['mean_absolute_error']) # compiling the model
model.summary() # model summary

# train the model
history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

#Access the training data 
training_hist = history.history


# evaluate model
train_loss, train_mean_absolute_error = model.evaluate(x_test, y_test)
print("\n \n")
print(f'Mean absolute error: {train_mean_absolute_error}')

#predict the model on test dataset
y_pred= model.predict(x_test)

# visualize the training history
training_loss = training_hist['loss']
training_val_loss = training_hist['val_loss']

training_epochs = range(1, len(training_loss)+1)

plt.plot(training_epochs, training_loss, 'bo', label='loss')
plt.plot(training_epochs, training_val_loss, 'b', label='val_loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()