import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
from scipy.stats import norm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

"""
These functions are used in the options trading strategy in round 4

The fit_vol function uses bisection search to find the implied volatility which
minimizes the difference between the black-scholes price and the market observed
price of coconut coupon

"""


def objective(df_coupon, df_coco, strike=1000, daily_vol=0.010103855204582212):
    df_coco['S/K'] = df_coco['mid_price'] / strike
    df_coupon['d1'] = (np.log(df_coco['S/K']) + daily_vol ** 2 * (250 - df_coupon['day']) / 2) / (
            daily_vol * np.sqrt(250 - df_coupon['day']))
    df_coupon['d2'] = (np.log(df_coco['S/K']) - daily_vol ** 2 * (250 - df_coupon['day']) / 2) / (
            daily_vol * np.sqrt(250 - df_coupon['day']))
    df_coupon['fair_value'] = df_coco['mid_price'] * norm.cdf(df_coupon['d1']) - strike * norm.cdf(df_coupon['d2'])
    return (df_coupon['mid_price'] - df_coupon['fair_value']).mean()


def fit_vol(df_coupon, df_coco, strike, lower_bound, upper_bound, tolerance):
    # Bisection loop
    while (upper_bound - lower_bound) > tolerance:
        midpoint = (lower_bound + upper_bound) / 2
        objective_midpoint = objective(df_coupon, df_coco, strike, midpoint)
        if objective_midpoint < 0:
            upper_bound = midpoint
        else:
            lower_bound = midpoint

    # Return the midpoint as the estimated implied volatility
    return (lower_bound + upper_bound) / 2


"""
The follow function is used to find the optimal trade & exit threshold for a mean version strategy
"""


def optimal_mean_reversion(df, col, mean, std):
    max = 0
    max_i = -1
    max_j = -1
    for i in np.arange(0.1, 1, 0.1):
        for j in np.arange(0.5, 2.25, 0.25):
            df['signal'] = df[col].transform(
                lambda x: 1 if x > mean + i * std else (-1 if x < mean - i * std * j else np.nan))
            df['signal'] = df['signal'].ffill()
            pnl = df['signal'].diff().abs().sum() * (i * std * (1 + j))
            if pnl > max:
                max = pnl
                max_i = i
                max_j = j

    print(f"Optimal trade threshold: {max_i}")
    print(f"Optimal exit threshold: -{max_j}")


"""
The follow functions are used in statistical analysis 
"""


def run_stats(df):
    products = df['product'].unique()
    dfs = {}

    for product in products:
        # Define paramters to run stats on
        cols = ['vwap_bid', 'vwap_ask', 'vwap_mid', 'spread', 'vwap_spread', 'imbalance']

        dfs[product] = df[df['product'] == product]
        print(f"Product: {product}")
        print(dfs[product][cols].describe().round(2))
        print("-------------------------------")


def test_stationary(df, d=1):
    df['diff'] = df['mid_price'].diff(d)

    # Assuming `series` is your Pandas Series containing the time series data
    adf_test = adfuller(df['diff'].dropna())

    print('ADF Statistic: %f' % adf_test[0])
    print('p-value: %f' % adf_test[1])
    print('Critical Values:')
    for key, value in adf_test[4].items():
        print('\t%s: %.3f' % (key, value))

    from statsmodels.tsa.stattools import kpss

    kpss_test = kpss(df['vwap_mid'].dropna(), nlags='auto')

    print('KPSS Statistic: %f' % kpss_test[0])
    print('p-value: %f' % kpss_test[1])
    print('Critical Values:')
    for key, value in kpss_test[3].items():
        print('\t%s: %.3f' % (key, value))


def test_seasonal(df, d=1, period=1000):
    # Decompose the series to observe trend, seasonality, and residuals
    decompose_result = seasonal_decompose(df['mid_price'].diff(d).dropna(), model='additive',
                                          period=period)  # Adjust 'period' based on your data
    decompose_result.plot()
    plt.show()


def test_acf_pacf(df, d=1):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    plot_acf(df['vwap_mid'].diff(d).dropna(), lags=50, ax=ax1, )
    plot_pacf(df['vwap_mid'].diff(d).dropna(), lags=50, ax=ax2, )
    plt.show()


def test_arima(df):
    # Load your dataset
    df = df['vwap_mid']
    split_point = int(len(df) * 0.7)
    train, test = df[:split_point], df[split_point:]

    # Optionally, plot to see the split
    plt.figure(figsize=(4, 3))
    plt.plot(train, label='Training set')
    plt.plot(test, label='Test set')
    plt.legend()
    plt.show()

    # Fit the model on the training set
    model = ARIMA(train, order=(2, 2, 1))
    model_fit = model.fit()

    # Summary of the model
    print(model_fit.summary())

    # Forecast
    forecast = model_fit.forecast(steps=len(test))

    # Plotting the forecast vs actual values
    plt.figure(figsize=(8, 5))
    plt.plot(train.index, train, label='Training Data')
    plt.plot(test.index, test, label='Actual Value')
    plt.plot(test.index, forecast, label='Forecasted Value', color='red')
    plt.legend()
    plt.show()


"""
The prices_df function reads the CSV files into dataframe and 
calculate the vwap midprice as well as order imbalance for a particular day/round

The trades_df function summarizes the trades in the CSV files

The parse_log_file function processes the log files returned by IMC after each submission
"""


def prices_df(day, directory_path='./data/', file_prefix='prices_round_'):
    # Dictionary to hold DataFrames
    df = pd.DataFrame()
    for day_index in range(day, day + 1):
        for index in range(day - 3, day):
            file_name = directory_path + file_prefix + f'{day_index}' + f'_day_{index}.csv'
            print(f"Reading {file_name}...")
            temp_df = pd.read_csv(file_name, delimiter=';')

            # Fill NA
            temp_df.fillna(0, inplace=True)

            # Make sure these columns are numeric
            exclude = ['product']
            cols = [col for col in temp_df.columns if col not in exclude]
            temp_df[cols] = temp_df[cols].apply(pd.to_numeric, errors='coerce')

            # Concatenate
            df = pd.concat([df, temp_df], axis=0)

    # Reset index
    df = df.reset_index(drop=True)

    # Calculate VWAPs
    total_volume_bid = df['bid_volume_1'] + df['bid_volume_2'] + df['bid_volume_3']
    total_volume_ask = df['ask_volume_1'] + df['ask_volume_2'] + df['ask_volume_3']
    df['vwap_bid'] = (df['bid_price_1'] * df['bid_volume_1'] + df['bid_price_2'] * df['bid_volume_2'] + df[
        'bid_price_3'] * df['bid_volume_3']) / total_volume_bid
    df['vwap_ask'] = (df['ask_price_1'] * df['ask_volume_1'] + df['ask_price_2'] * df['ask_volume_2'] + df[
        'ask_price_3'] * df['ask_volume_3']) / total_volume_ask
    df['vwap_mid'] = (df['vwap_ask'] + df['vwap_bid']) / 2
    df['vwap_spread'] = df['vwap_ask'] - df['vwap_bid']
    df['spread'] = df['ask_price_1'] - df['bid_price_1']

    # Calculate book imbalance
    df['imbalance'] = (df['bid_volume_1'] - df['ask_volume_1']) / (df['bid_volume_1'] + df['ask_volume_1'])

    return df


def trades_df(day, directory_path='./data/', file_prefix='trades_round_'):
    # Dictionary to hold DataFrames
    df = pd.DataFrame()
    for day_index in range(day, day + 1):
        for index in range(day - 3, day):
            file_name = directory_path + file_prefix + f'{day_index}' + f'_day_{index + day_index - 1}_nn.csv'
            print(f"Reading {file_name}...")
            temp_df = pd.read_csv(file_name, delimiter=';')

            # Fill NA
            temp_df.fillna(0, inplace=True)

            # Make sure these columns are numeric
            exclude = ['symbol', 'buyer', 'seller', 'currency']
            cols = [col for col in temp_df.columns if col not in exclude]
            temp_df[cols] = temp_df[cols].apply(pd.to_numeric, errors='coerce')

            # Concatenate
            df_trades = pd.concat([df, temp_df], axis=0)

    # Reset index
    df_trades = df_trades.reset_index(drop=True)

    return df_trades


def find_latest_final_log_file(directory_path='./log_data/'):
    # Filter files to include only those with ".log" extension and containing "final" in the filename
    files = [os.path.join(directory_path, f) for f in os.listdir(directory_path)
             if os.path.isfile(os.path.join(directory_path, f))
             and f.endswith('.log') and f != '.DS_Store' and 'final' in f.lower()]

    if files:
        return max(files, key=os.path.getmtime)
    return None


def parse_log_file(log_file_path):
    with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as file:
        lines = file.readlines()

    activities_data = []
    trade_history_json_str = ""
    process_lines = False
    in_activities_section = False
    awaiting_activities_columns = False  # Flag to indicate next line should define columns
    activities_columns = []

    for line in lines:
        line = line.strip()
        if 'Activities log' in line:
            process_lines = True
            in_activities_section = True
            awaiting_activities_columns = True  # Next line should be columns
            continue
        elif 'Trade History' in line:
            in_activities_section = False
            trade_history_json_str += "["  # Start capturing trade history JSON
            continue

        if process_lines:
            if in_activities_section and awaiting_activities_columns:
                activities_columns = line.split(';')  # Set columns from the next line
                awaiting_activities_columns = False  # Reset flag after setting columns
            elif in_activities_section and not awaiting_activities_columns:
                activities_data.append(line.split(';'))

            elif line and not line.startswith('[') and not line.endswith(']'):
                # Ensure proper JSON format for trade history entries
                trade_history_json_str += line

    # Correctly close the trade history JSON string
    if not trade_history_json_str.endswith(']'):
        trade_history_json_str += ']'

    # Create the DataFrames
    df_activities = pd.DataFrame(activities_data, columns=activities_columns) if activities_columns else pd.DataFrame()

    try:
        trade_history_data = json.loads(trade_history_json_str)
        df_trade_history = pd.DataFrame(trade_history_data)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON for Trade History: {e}")
        df_trade_history = pd.DataFrame()  # Return an empty DataFrame in case of error

    # Drop empty bid & offer entries
    # Need to think about how empty columns affect our calculation
    # Drop rows where all elements are NaN
    df_activities = df_activities.dropna(subset=['ask_price_1', 'bid_price_1'])

    # Make sure these columns are numeric
    exclude = ['product', 'buyer', 'seller', 'symbol', 'currency']
    cols = [col for col in df_activities.columns if col not in exclude]
    df_activities[cols] = df_activities[cols].apply(pd.to_numeric, errors='coerce')

    return df_activities, df_trade_history
