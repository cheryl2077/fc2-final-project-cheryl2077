import yfinance as yf
import pandas as pd

def load_price_data(tickers, start="2019-01-01", end="2024-01-01"):

    # Download adjusted close prices for Zuser-selected tickers.

    df = yf.download(tickers, start=start, end=end, auto_adjust=False)

    # utilized AI to debug and generated this multiIndex case

    # Case 1: MultiIndex columns (multiple tickers)
    if isinstance(df.columns, pd.MultiIndex):
        # Check level 0 contains "Adj Close"
        if "Adj Close" in df.columns.levels[0]:
            df = df["Adj Close"]
        else:
            raise KeyError("Downloaded data has no 'Adj Close' field for multi-ticker input.")

    # Case 2: Single ticker (regular DataFrame)
    else:
        if "Adj Close" in df.columns:
            df = df[["Adj Close"]]
            df.columns = tickers  # rename to ticker name
        else:
            raise KeyError("Downloaded data has no 'Adj Close' column for single ticker input.")

    return df.dropna()



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_log_returns(price_df):
    # Return log price differences
    log_prices = np.log(price_df)
    return log_prices.diff().dropna()

def compute_correlation_matrix(returns_df):
    # Return correlation matrix of returns
    return returns_df.corr()