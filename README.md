
# Stock Buy Signal Prediction using LSTM Neural Network

This project demonstrates how to predict buy signals in stock trading using a Long Short-Term Memory (LSTM) neural network. The example uses historical stock price data to train the model and predict future buy signals.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Data Preparation](#data-preparation)
- [Data Fetching](#data-fetching)
- [Data Transformation using pandas_ta](#data-transformation-using-pandas_ta)
- [Why Use LSTM for Stock Prediction](#why-use-lstm-for-stock-prediction)
- [Model Training](#model-training)
- [Prediction](#prediction)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The aim of this project is to build a neural network model that can predict buy signals for stocks based on historical price data. The model uses an LSTM network, which is well-suited for time series data due to its ability to capture temporal dependencies.

## Installation

To get started, clone this repository and install the necessary dependencies.

```bash
git clone https://github.com/your-username/stock-buy-signal-prediction.git
cd stock-buy-signal-prediction
pip install numpy pandas tensorflow pandas_ta requests
```

## Usage

1. **Prepare your data**: Ensure you have historical stock price data in CSV format.
2. **Run the script**: Follow the steps in the script to preprocess data, train the model, and make predictions.

## Data Preparation

1. **Load your data**: Load the historical stock price data from a CSV file.
2. **Normalize the data**: Use MinMaxScaler to normalize the data for better model performance.
3. **Create the dataset**: Prepare the dataset with features and labels for training.

Example code:

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load your data
data = pd.read_csv('your_stock_data.csv')

# Using 'Close' prices for simplicity
prices = data['Close'].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)

# Create a dataset with features and labels
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 60
X, Y = create_dataset(scaled_prices, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)  # LSTM expects input shape: (samples, time steps, features)
```

## Data Fetching

To fetch historical stock price data, we use the Alpha Vantage API. Alpha Vantage provides free APIs for real-time and historical market data of various assets, including stocks, forex, and cryptocurrencies.

### Getting Started with Alpha Vantage

1. **Sign Up**: Create a free account on [Alpha Vantage](https://www.alphavantage.co/).
2. **API Key**: After signing up, you will receive an API key to access the data.
3. **Fetch Data**: Use the API key to fetch historical stock data. Below is an example code to fetch data using Alpha Vantage API:

```python
import requests
import pandas as pd

api_key = 'YOUR_API_KEY'
symbol = 'IBM'
url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&outputsize=full&apikey={api_key}'

response = requests.get(url)
data = response.json()

# Convert the JSON data to a pandas DataFrame
df = pd.DataFrame(data['Time Series (Daily)']).T
df = df.rename(columns={
    '1. open': 'open',
    '2. high': 'high',
    '3. low': 'low',
    '4. close': 'close',
    '5. adjusted close': 'adjusted_close',
    '6. volume': 'volume',
    '7. dividend amount': 'dividend_amount',
    '8. split coefficient': 'split_coefficient'
})
df.index = pd.to_datetime(df.index)
df = df.astype(float)
print(df.head())
```

## Data Transformation using pandas_ta

After fetching the historical stock data, we use the `pandas_ta` library for technical analysis and data transformation. Specifically, we calculate the Simple Moving Averages (SMA) for 20, 50, and 200 days.

### Example Code for SMA Calculation

```python
import pandas_ta as ta

# Calculate SMA
df['SMA_20'] = ta.sma(df['close'], length=20)
df['SMA_50'] = ta.sma(df['close'], length=50)
df['SMA_200'] = ta.sma(df['close'], length=200)

print(df[['close', 'SMA_20', 'SMA_50', 'SMA_200']].tail())
```

## Why Use LSTM for Stock Prediction

Long Short-Term Memory (LSTM) networks are a type of recurrent neural network (RNN) capable of learning long-term dependencies. They are well-suited for time series data because they can remember past information for long periods. This makes LSTM an excellent choice for stock price prediction, where historical price data can influence future prices.

### Advantages of LSTM for Stock Prediction

1. **Memory**: LSTMs can remember previous inputs, which helps in understanding the temporal dependencies in stock prices.
2. **Handling Sequence Data**: LSTM is designed to handle sequence data, making it ideal for time series prediction.
3. **Avoiding the Vanishing Gradient Problem**: LSTM networks use mechanisms like forget gates to prevent issues like the vanishing gradient problem, which can occur in traditional RNNs.

By leveraging LSTM networks, we aim to capture the intricate patterns and trends in stock prices to predict buy signals accurately.

## Model Training

1. **Define the LSTM model**: Create an LSTM model using Keras.
2. **Compile the model**: Compile the model with the Adam optimizer and mean squared error loss.
3. **Train the model**: Train the model using the training dataset.

Example code:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Split the data into training and testing sets
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
Y_train, Y_test = Y[:split], Y[split:]

model.fit(X_train, Y_train, batch_size=1, epochs=1)
```

## Prediction

1. **Make predictions**: Use the trained model to make predictions on the test dataset.
2. **Generate buy signals**: Implement simple logic to generate buy signals based on predictions.

Example code:

```python
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Example logic for a simple buy signal based on prediction
buy_signals = []
for i in range(1, len(predictions)):
    if predictions[i] > predictions[i - 1]:
        buy_signals.append(True)  # Buy signal
    else:
        buy_signals.append(False)  # No buy signal

# Print the buy signals
print(buy_signals)
```

## Contributing

Contributions are welcome! Please fork this repository and submit a pull request for any improvements or additions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
