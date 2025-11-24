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

def main():
    
    apple = yf.Ticker("AAPL")

    prices = apple.history(period="5y")

    print(prices.tail())
if __name__ == "__main__":
    main()
