import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from collections import deque
from matplotlib.lines import Line2D

import requests
from datetime import datetime
from time import time
from datetime import timedelta


# ----------------- Adding Function Based on Higher Highs , ....
# Function for each point situation
def getHigherLows(data: np.array, order=5, K=2):
    '''
    Finds consecutive higher lows in price pattern.
    Must not be exceeded within the number of periods indicated by the width
    parameter for the value to be confirmed.
    K determines how many consecutive lows need to be higher.
    '''
    # Get lows
    low_idx = argrelextrema(data, np.less, order=order)[0]
    lows = data[low_idx]
    # Ensure consecutive lows are higher than previous lows
    extrema = []
    ex_deque = deque(maxlen=K)
    for i, idx in enumerate(low_idx):
        if i == 0:
            ex_deque.append(idx)
            continue
        if lows[i] < lows[i - 1]:
            ex_deque.clear()

        ex_deque.append(idx)
        if len(ex_deque) == K:
            extrema.append(ex_deque.copy())

    return extrema


def getLowerHighs(data: np.array, order=5, K=2):
    '''
    Finds consecutive lower highs in price pattern.
    Must not be exceeded within the number of periods indicated by the width
    parameter for the value to be confirmed.
    K determines how many consecutive highs need to be lower.
    '''
    # Get highs
    high_idx = argrelextrema(data, np.greater, order=order)[0]
    highs = data[high_idx]
    # Ensure consecutive highs are lower than previous highs
    extrema = []
    ex_deque = deque(maxlen=K)
    for i, idx in enumerate(high_idx):
        if i == 0:
            ex_deque.append(idx)
            continue
        if highs[i] > highs[i - 1]:
            ex_deque.clear()

        ex_deque.append(idx)
        if len(ex_deque) == K:
            extrema.append(ex_deque.copy())

    return extrema


def getHigherHighs(data: np.array, order=5, K=2):
    '''
    Finds consecutive higher highs in price pattern.
    Must not be exceeded within the number of periods indicated by the width
    parameter for the value to be confirmed.
    K determines how many consecutive highs need to be higher.
    '''
    # Get highs
    high_idx = argrelextrema(data, np.greater, order=5)[0]
    highs = data[high_idx]
    # Ensure consecutive highs are higher than previous highs
    extrema = []
    ex_deque = deque(maxlen=K)
    for i, idx in enumerate(high_idx):
        if i == 0:
            ex_deque.append(idx)
            continue
        if highs[i] < highs[i - 1]:
            ex_deque.clear()

        ex_deque.append(idx)
        if len(ex_deque) == K:
            extrema.append(ex_deque.copy())

    return extrema


def getLowerLows(data: np.array, order=5, K=2):
    '''
    Finds consecutive lower lows in price pattern.
    Must not be exceeded within the number of periods indicated by the width
    parameter for the value to be confirmed.
    K determines how many consecutive lows need to be lower.
    '''
    # Get lows
    low_idx = argrelextrema(data, np.less, order=order)[0]
    lows = data[low_idx]
    # Ensure consecutive lows are lower than previous lows
    extrema = []
    ex_deque = deque(maxlen=K)
    for i, idx in enumerate(low_idx):
        if i == 0:
            ex_deque.append(idx)
            continue
        if lows[i] > lows[i - 1]:
            ex_deque.clear()

        ex_deque.append(idx)
        if len(ex_deque) == K:
            extrema.append(ex_deque.copy())

    return extrema


# --------------------- Getting Price From API
base_url = "https://api.kucoin.com"
coin_pair = "BTC-USDT"  # BTC-USDT
frequency = "1hour"  # 1hour 4hour 1min
# get timestamp date of today in seconds
now_is = int(time())
days = 5
# sec  min  hour days
days_delta = 60 * 60 * 24 * days
start_At = now_is - days_delta
# print(now_is)
price_url = f"/api/v1/market/candles?type={frequency}&symbol={coin_pair}&startAt={start_At}&endAt={now_is}"

price_dict = {}

prices = requests.get(base_url + price_url).json()
print(prices)

for item in prices['data']:
    # convert date from timestamp to Y M D
    date_converted = datetime.fromtimestamp(int(item[0])).strftime("%Y-%m-%d-%H")
    price_dict[date_converted] = item[2]

priceDF = pd.DataFrame(price_dict, index=["price"]).T
# Convert prices into a float
priceDF['price'] = priceDF['price'].astype(float)

# convert dates to datetime from object
priceDF.index = pd.to_datetime(priceDF.index)

# reverse dates
priceDF = priceDF.iloc[::-1]
data = priceDF
# -----------------------------

# ----------------------------  Calculate HH LH,......
close = data['price'].values
dates = data.index

order = 3
K = 2

hh = getHigherHighs(close, order, K)
hl = getHigherLows(close, order, K)
ll = getLowerLows(close, order, K)
lh = getLowerHighs(close, order, K)

# --------------- Calculate RSI
# ----------------------------------- Calculate RSI
def calcRSI(data, P=14):
    data['diff_close'] = data['price'] - data['price'].shift(1)
    data['gain'] = np.where(data['diff_close'] > 0, data['diff_close'], 0)
    data['loss'] = np.where(data['diff_close'] < 0, np.abs(data['diff_close']), 0)
    data[['init_avg_gain', 'init_avg_loss']] = data[
        ['gain', 'loss']].rolling(P).mean()
    avg_gain = np.zeros(len(data))
    avg_loss = np.zeros(len(data))
    for i, _row in enumerate(data.iterrows()):
        row = _row[1]
        if i < P - 1:
            last_row = row.copy()
            continue
        elif i == P - 1:
            avg_gain[i] += row['init_avg_gain']
            avg_loss[i] += row['init_avg_loss']
        else:
            avg_gain[i] += ((P - 1) * avg_gain[i - 1] + row['gain']) / P
            avg_loss[i] += ((P - 1) * avg_loss[i - 1] + row['loss']) / P

        last_row = row.copy()

    data['avg_gain'] = avg_gain
    data['avg_loss'] = avg_loss
    data['RS'] = data['avg_gain'] / data['avg_loss']
    data['RSI'] = 100 - 100 / (1 + data['RS'])
    return data


data_rsi = calcRSI(data.copy())
rsi = data_rsi['RSI'].values

