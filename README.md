****Comparative Study of Stock Market Forecasting Using Time Series Models****


**Description:**
Data Science final year project on "Stock market prediction using Istanbul stock exchange (ISE) dataset from UCl dataset repository.

This project involves a comparative analysis of stock market forecasting using three different time series models: Long Short-Term Memory (LSTM), AutoRegressive Integrated Moving Average (ARIMA), and Facebook's Prophet. The analysis is conducted using the Istanbul Stock Market dataset, which contains financial indicators from various international markets. The goal of this project is to predict the Istanbul Stock Exchange (ISE) index and compare the performance of the models using various evaluation metrics.

The dataset consists of 7 columns: ISE, SP, DAX, FTSE, NIKKEI, BOVESPA, EU, and EM, where ISE is the target variable. The data is preprocessed and normalized before being fed into the models. The dataset is split into 80% for training and 20% for testing, and the models are evaluated using Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R²) metrics.

**Code File:**
Avaliable in both .ipynb (using colab) and .py(using python file)
**File name:**
**Using Colab:** ds_final_proj_stock_market_colab.ipynb

**Python file:** stock_market_prediction_ds.py

Before running this code, Install the below python packages and run the code.

## Installation

To run this project, you will need Python 3.8+ and the following packages:

- `ucimlrepo` for fetching the dataset
- `pmdarima` for the ARIMA model
- `prophet` for the Facebook Prophet model
- `numpy`, `pandas`, `scikit-learn`, `tensorflow`, `matplotlib`, `seaborn`

Install the required packages using pip:

**Fetch the Dataset:**
from ucimlrepo import fetch_ucirepo
dataset = fetch_ucirepo(id=247)

**Prepare the Data:**
Ensure the dataset is properly preprocessed and normalized.
The dataset should contain the following columns: ISE, SP, DAX, FTSE, NIKKEI, BOVESPA, EU, EM, with ISE as the target variable

**Models Used**
**LSTM (Long Short-Term Memory)**
LSTM is a type of Recurrent Neural Network (RNN) effective for time series prediction due to its ability to remember previous data points over long sequences. It is particularly useful for modeling sequential data and capturing temporal dependencies.

**ARIMA (AutoRegressive Integrated Moving Average)**
ARIMA is a well-known statistical method for time series forecasting. It combines autoregression, differencing, and moving average components to model time-dependent data.

**Prophet**
Prophet is an open-source forecasting tool developed by Facebook, designed to handle seasonality in time series data and provide interpretable results.


**Data**
The dataset used in this project is the Istanbul Stock Market dataset, which includes financial indicators from various international markets. The dataset contains 7 columns:
ISE (target variable), SP, DAX, FTSE, NIKKEI, BOVESPA, EU, EM
Data preprocessing and normalization are performed before feeding the data into the models. The dataset is split into 80% for training and 20% for testing.

**Training and Testing**
All models were trained and tested using the same dataset split (80% for training, 20% for testing). Various learning rates, timesteps, epochs, and batch sizes were used to optimize the model performance. The following metrics were used for evaluation:
  Mean Absolute Error (MAE)
  Mean Squared Error (MSE)
  Root Mean Squared Error (RMSE)
  R-squared (R²)
  Results

**Summary of Model Performance:**
 ** LSTM: **Provided the best overall performance before feature extraction, with a training loss as low as 0.0100. However, slight overfitting was observed after feature extraction.
  **ARIMA**: Showed good performance with the data, but occasional overfitting was noticed both before and after feature extraction.
  **Prophet:** Exhibited more overfitting than the other models, although it achieved lower RMSE values.
  
Overall, LSTM was the most effective model, especially before feature extraction, despite some overfitting after feature extraction.
