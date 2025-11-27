# Credit Card Fraud Detection Script
# Copyright 2025 Damon Chase
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, this file
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS
# OF ANY KIND, either express or implied.

import pandas as pd
import yfinance as yf
import numpy as np
from ta.momentum import RSIIndicator

def main():
    
    # Get Apple's stock data for past month
    apple = yf.Ticker("AAPL")
    prices = apple.history(period="3y")

    # New dataframe with features and label
    df = pd.DataFrame(prices)

    # Feature engineering: 10 day moving averages
    start, stop = 0, 10
    ma_dict = {}
    while stop < len(df["Close"]):
        window = df["Close"].iloc[start:stop]
        index = df["Close"].index[stop]
        avg = np.mean(window)
        ma_dict.update({index: int(avg)})
        start += 1; stop += 1
    df["ma"] = pd.Series(ma_dict)

    # Feature engineering: RSI feature
    df["RSI"] = RSIIndicator(df["Close"], 14, False).rsi()

    # Feature engineering: Next days closing price (target)
    nc_dict = {}
    for i in range(0, len(df["Close"]) - 1):
        key = df["Close"].index[i + 1]
        value = df["Close"].iloc[i + 1]
        nc_dict.update({key: int(value)})
    df["NextClose"] = pd.Series(nc_dict)

if __name__ == "__main__":
    main()
