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
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

def main():
    
    # Get Dell's stock data for past month
    dell = yf.Ticker("DELL")
    prices = dell.history(period="10y")

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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, shuffle=False)

    # Instantiate and train model
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        n_jobs=-1,
        reg_lambda=1.0
    )
    model.fit(X_train, y_train)

    # Test model
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE: {rmse}")

    # Plot predicted vs actual
    dates = y_test.index.to_numpy()
    plt.plot(dates, y_pred, label="Prediction")
    plt.plot(dates, y_test, label="Actual")
    plt.xlabel("Dates")
    plt.ylabel("Price in USD")
    plt.title("Actual vs Predicted Next Day Closing Price of Dell Stock")
    plt.legend()
    plt.show()

    # Real Use Case: Predict a next day closing price
    ts = pd.to_datetime("2025-11-25 00:00:00-05:00")
    print(f"Actual Closing Price on Wednesday November 26th: {df["Close"].loc[ts]}")
    data = [[123.089996, 127.120003, 123.050003, 125.919998, 126.601001]]
    print(f"Predicted Closing Price for Wendesday November 26th: {model.predict(data)}")

if __name__ == "__main__":
    main()
