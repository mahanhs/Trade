import requests
from datetime import datetime
from time import time
import pandas as pd

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

print(priceDF)
