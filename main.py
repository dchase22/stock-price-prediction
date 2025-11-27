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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def main():
    
    # Get Apple's stock data for past month
    apple = yf.Ticker("AAPL")
    prices = apple.history(period="5y")

    # New dataframe with features and label
    df = pd.DataFrame(prices)

    # Feature engineering: Remove un-important columns for prediction
    df = df.drop(["Dividends", "Stock Splits", "Volume"], axis=1)

    # Feature engineering: 10 day moving averages
    df["MA"] = df["Close"].rolling(window=10).mean()

    # Feature engineering: Next days closing price (target)
    df["NextClose"] = df["Close"].shift(-1)

    # Clean data: Get rid of rows with empty vals
    df.dropna(inplace=True)

    # Split train and test data maintaining chronological order
    X = df.drop("NextClose", axis=1)
    y = df["NextClose"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Instantiate and train model
    model = RandomForestRegressor(n_estimators=300, n_jobs=-1)
    model.fit(X_train, y_train)

    # Test model
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE: {rmse}")
        
if __name__ == "__main__":
    main()
