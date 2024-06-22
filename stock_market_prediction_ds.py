# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 11:03:02 2024

@author: Raghavendhra Devineni
"""

'''
Installing packages

Install the ucimlrepo package
    pip install ucimlrepo
    

'''

from ucimlrepo import fetch_ucirepo 
import matplotlib.pyplot as plt
import pandas as pd
  
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
print(stock_data.head())

# printing the summary statistics
# Descriptive statistics include those that summarize the central tendency, dispersion, and shape of a dataset’s distribution, excluding NaN values
print(stock_data.describe())


# information about the dataset
print(stock_data.info())

# print("\n","Data Columns: ", data_info.columns)

# print("varibles data :")
print(ise_data.variables)


# convert the date datatpe to datetime
print(stock_data['date'].head())

stock_data['date'] = pd.to_datetime(stock_data['date'], format='%d-%b-%y')
print(stock_data['date'].head())
print(stock_data.dtypes)

# set the date column as Index
stock_data.set_index('date', inplace=True)

#print few rows
print(stock_data.head())

# check for null values in the dataset
data_null_values = stock_data.isnull().sum()
print("null values found in dataset: ", data_null_values)

# calculating the mean of each month and restructure the data
stock_data_mean = stock_data.resample('M').mean()
print(stock_data_mean.head())

# Plot each column
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10), constrained_layout=True)
data_columns = stock_data.columns
for col, ax in enumerate(axes.flat):
    if col < len(data_columns):
        stock_data_mean[data_columns[col]].plot(ax=ax)
        ax.set_title(data_columns[col])
        
plt.show()

