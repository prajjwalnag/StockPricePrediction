#Extract data from api and store in json format 
import requests
import argparse
import json
import os

API_KEY = 'N4BKPEV09F3A1KRH'
BASE_URL = 'https://www.alphavantage.co/query'

def extract_stock_data(symbol, function='TIME_SERIES_DAILY', interval=None):
    params = {
        'function': function,
        'symbol': symbol,
        'apikey': API_KEY,
        'outputsize': 'full',
        'datatype': 'json'
    }
    if function == 'TIME_SERIES_INTRADAY' and interval:
        params['interval'] = interval
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    if 'Error Message' in data:
        raise ValueError(f"Error fetching data for {symbol}: {data['Error Message']}")
    if 'Note' in data:
        raise ValueError(f"API call frequency limit reached: {data['Note']}")
    return data

def save_data_to_json(data, symbol, filename):
    loc_path='stock/'+symbol
    os.makedirs(loc_path, exist_ok=True)
    filepath = os.path.join(loc_path, filename)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Data for {symbol} saved to {filepath}")

def main():
    parser = argparse.ArgumentParser(description='Extract stock data from Alpha Vantage and save as JSON.')
    parser.add_argument('symbol', type=str, help='Stock symbol to extract data for')
    parser.add_argument('--function', type=str, choices=['daily', 'intraday', 'weekly', 'monthly'], default='daily', help='Time series function (daily, intraday, weekly, monthly)')
    parser.add_argument('--interval', type=str, choices=['1min', '5min', '15min', '30min', '60min'], help='Interval for intraday data')
    parser.add_argument('--output', type=str, default='stock_data.json', help='Output JSON file name')
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
        raw_data = extract_stock_data(args.symbol, function, interval)
        save_data_to_json(raw_data, args.symbol, args.output)
    except ValueError as e:
        print(e)

if __name__ == '__main__':
    main()
