# scripts/predict_eth_price.py

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the data
eth_data = pd.read_csv('data/prices.csv', parse_dates=['Date'], index_col='Date')

# Handle missing values by forward filling
eth_data.fillna(method='ffill', inplace=True)

# Use only the 'Close' price for modeling
eth_close = eth_data['Close']

# Splitting the data into training and testing sets
train_size = int(len(eth_close) * 0.8)
train, test = eth_close[:train_size], eth_close[train_size:]

# Fit ARIMA model
model = ARIMA(train, order=(5, 1, 0))  # Adjust parameters based on your analysis
model_fit = model.fit()

# Print the summary of the model
print(model_fit.summary())

# Forecasting the test set
start_index = len(train)
end_index = len(train) + len(test) - 1
predictions = model_fit.predict(start=start_index, end=end_index, typ='levels')

# Evaluate the model
rmse = np.sqrt(mean_squared_error(test, predictions))
print(f'Root Mean Square Error: {rmse}')

# Plotting the results
plt.figure(figsize=(14, 7))
plt.plot(train, label='Train')
plt.plot(test, label='Test')
plt.plot(predictions, label='Predicted')
plt.title('Ethereum Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()
