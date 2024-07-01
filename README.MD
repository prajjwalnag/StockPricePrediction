
# Stock Buy Signal Prediction using LSTM Neural Network

This project demonstrates how to predict buy signals in stock trading using a Long Short-Term Memory (LSTM) neural network. The example uses historical stock price data to train the model and predict future buy signals.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Data Preparation](#data-preparation)
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
pip install numpy pandas tensorflow
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
