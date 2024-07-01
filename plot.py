import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import mplfinance as mpf
import mplcursors
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import os

def read_csv_data(filepath):
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    df.sort_index(inplace=True)
    return df

def plot_stock_data(df, symbol, function):
    mc = mpf.make_marketcolors(up='green', down='red', wick={'up': 'green', 'down': 'red'}, edge={'up': 'green', 'down': 'red'}, volume='in')
    s = mpf.make_mpf_style(marketcolors=mc, gridstyle=':', gridcolor='gray', figcolor='black', facecolor='black')

    addplot = [mpf.make_addplot(df['SMA20'], color='lime', width=1.5),
               mpf.make_addplot(df['SMA50'], color='cyan', width=1.5),
               mpf.make_addplot(df['SMA200'], color='magenta', width=1.5)]

    fig, axlist = mpf.plot(df, type='candle', style=s, title=f'{symbol} ({function})', ylabel='Price', volume=True, show_nontrading=True, addplot=addplot, returnfig=True)
    mplcursors.cursor(axlist[0].get_lines()).connect("add", lambda sel: sel.annotation.set_text(f'{df.index[sel.target.index]}: {sel.target}'))
    plt.show()

def prepare_data(df):
    # Prepare features and target for the next 30 days
    df['Target'] = df['Close'].shift(-30)
    df.dropna(inplace=True)

    # Additional features
    df['High-Low'] = df['High'] - df['Low']
    df['Close-Open'] = df['Close'] - df['Open']

    # Features and target
    features = df[['Close', 'SMA20', 'SMA50', 'SMA200', 'High-Low', 'Close-Open']]
    target = df['Target']

    # Scale the features
    scaler = MinMaxScaler(feature_range=(0, 1))
    features_scaled = scaler.fit_transform(features)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.35, shuffle=False)
    
    # Reshape data for LSTM [samples, time steps, features]
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    
    return X_train, X_test, y_train, y_test, scaler

def build_and_train_nn(X_train, y_train, X_test, y_test):
    # Build the neural network model
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    history = model.fit(X_train, y_train, epochs=100, batch_size=8, validation_data=(X_test, y_test), verbose=1)

    return model, history

def plot_predictions(predictions, y_test, df, scaler, shift_days=30):
    # Inverse transform the predictions and actual values
    inverse_pred = np.zeros((len(predictions), scaler.data_max_.shape[0]))
    inverse_pred[:, -1] = predictions[:, 0]
    predictions = scaler.inverse_transform(inverse_pred)[:, -1]

    inverse_y_test = np.zeros((len(y_test), scaler.data_max_.shape[0]))
    inverse_y_test[:, -1] = y_test.values
    y_test = scaler.inverse_transform(inverse_y_test)[:, -1]

    # Shift the predictions to the left by 'shift_days'
    predictions_shifted = np.roll(predictions, -shift_days)
    predictions_shifted[-shift_days:] = np.nan  # Set the last 'shift_days' values to NaN

    plt.figure(figsize=(14, 7))
    plt.title('Stock Price Prediction', color='white')
    plt.xlabel('Date', color='white')
    plt.ylabel('Price', color='white')
    plt.plot(df.index[-len(y_test):], y_test, label='Actual Prices', color='blue')
    plt.plot(df.index[-len(predictions_shifted):], predictions_shifted, label='Predicted Prices', color='red')
    legend = plt.legend()
    for text in legend.get_texts():
        text.set_color("white")
    plt.gca().tick_params(axis='x', colors='white')
    plt.gca().tick_params(axis='y', colors='white')
    plt.gca().spines['bottom'].set_color('white')
    plt.gca().spines['top'].set_color('white')
    plt.gca().spines['left'].set_color('white')
    plt.gca().spines['right'].set_color('white')
    plt.show()

    # Print Mean Absolute Error excluding the NaN values
    valid_indices = ~np.isnan(predictions_shifted)
    mae = mean_absolute_error(y_test[valid_indices], predictions_shifted[valid_indices])
    print(f"Mean Absolute Error: {mae}")

# Modify main to pass the shift_days parameter
def main():
    parser = argparse.ArgumentParser(description='Read stock data from CSV, process, train neural network, and predict future prices.')
    parser.add_argument('input', type=str, help='Input CSV file name')
    args = parser.parse_args()
    
    try:
        stock_df = read_csv_data(args.input)
        print(stock_df.head())
        
        symbol = os.path.basename(args.input).split('_')[0]
        plot_stock_data(stock_df, symbol, "daily")

        X_train, X_test, y_train, y_test, scaler = prepare_data(stock_df)
        model, history = build_and_train_nn(X_train, y_train, X_test, y_test)
        
        # Predict future prices
        predictions = model.predict(X_test)
        
        # Shift the predictions 30 days earlier
        plot_predictions(predictions, y_test, stock_df, scaler, shift_days=30)
        
    except ValueError as e:
        print(e)

if __name__ == '__main__':
    main()
