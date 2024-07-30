

import yfinance as yf
import pandas as pd

def download_eth_data():
    # Download Ethereum data
    eth_data = yf.download('ETH-USD', start='2015-01-01', end='2024-01-01')
    # Ensure the data directory exists
    eth_data.to_csv('data/prices.csv')

if __name__ == '__main__':
    download_eth_data()
