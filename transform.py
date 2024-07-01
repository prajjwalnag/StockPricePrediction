import pandas as pd
import pandas_ta as ta
import argparse
import json
import os

def transform_data_to_dataframe(data, function, interval=None):
    if function == 'TIME_SERIES_DAILY':
        key = 'Time Series (Daily)'
    elif function == 'TIME_SERIES_WEEKLY':
        key = 'Weekly Time Series'
    elif function == 'TIME_SERIES_MONTHLY':
        key = 'Monthly Time Series'
    elif function == 'TIME_SERIES_INTRADAY' and interval:
        key = f'Time Series ({interval})'
    else:
        raise ValueError('Unsupported function type')
    
    if key not in data:
        print(f"Raw response: {data}")
        raise ValueError(f"Data not found for the requested function: {key}")

    time_series = data[key]
    df = pd.DataFrame.from_dict(time_series, orient='index')
    df = df.apply(pd.to_numeric)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    df.rename(columns={
        '1. open': 'Open',
        '2. high': 'High',
        '3. low': 'Low',
        '4. close': 'Close',
        '5. volume': 'Volume'
    }, inplace=True)
    return df

def calculate_sma(df, windows=[20, 50, 200]):
    for window in windows:
        df[f'SMA{window}'] = ta.sma(df['Close'], length=window)
    return df

def save_dataframe_to_csv(df, symbol, filename):
    os.makedirs(os.path.join('stock', symbol), exist_ok=True)
    filepath = os.path.join('stock', symbol, filename)
    df.to_csv(filepath)
    print(f"Dataframe for {symbol} saved to {filepath}")

def main():
    parser = argparse.ArgumentParser(description='Transform stock data from JSON to mplfinance format, calculate SMA with pandas-ta, and save as CSV.')
    parser.add_argument('input', type=str, help='Input JSON file name')
    parser.add_argument('--function', type=str, choices=['daily', 'intraday', 'weekly', 'monthly'], default='daily', help='Time series function (daily, intraday, weekly, monthly)')
    parser.add_argument('--interval', type=str, choices=['1min', '5min', '15min', '30min', '60min'], help='Interval for intraday data')
    parser.add_argument('--output', type=str, default='stock_data.csv', help='Output CSV file name')
    args = parser.parse_args()
    
    function_mapping = {
        'daily': 'TIME_SERIES_DAILY',
        'intraday': 'TIME_SERIES_INTRADAY',
        'weekly': 'TIME_SERIES_WEEKLY',
        'monthly': 'TIME_SERIES_MONTHLY'
    }

    function = function_mapping[args.function]
    interval = args.interval if args.function == 'intraday' else None

    try:
        with open(args.input, 'r') as f:
            raw_data = json.load(f)
        stock_df = transform_data_to_dataframe(raw_data, function, interval)
        stock_df = calculate_sma(stock_df)
        parts = args.input.split('/')
        symbol = parts[1]
        save_dataframe_to_csv(stock_df, symbol, args.output)
    except ValueError as e:
        print(e)

if __name__ == '__main__':
    main()

      