import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Download historical stock data from Yahoo Finance
ticker = 'AAPL'
data = yf.download(ticker, start="2020-01-01", end="2023-01-01")

# Display the first few rows of the data
data.head()
# Calculate the moving averages
short_window = 50
long_window = 200

data['Short_MA'] = data['Close'].rolling(window=short_window, min_periods=1).mean()
data['Long_MA'] = data['Close'].rolling(window=long_window, min_periods=1).mean()

# Display the moving averages
data[['Close', 'Short_MA', 'Long_MA']].tail()
# Create signals based on moving average crossovers
data['Signal'] = 0
data['Signal'][short_window:] = np.where(data['Short_MA'][short_window:] > data['Long_MA'][short_window:], 1, 0)

# Generate trading orders: Buy (1), Sell (-1)
data['Position'] = data['Signal'].diff()

# Display the buy/sell signals
data[['Close', 'Short_MA', 'Long_MA', 'Position']].tail()
# Plot the stock price with moving averages and buy/sell signals
plt.figure(figsize=(12,8))
plt.plot(data['Close'], label='Stock Price', alpha=0.5)
plt.plot(data['Short_MA'], label='50-day MA', alpha=0.7)
plt.plot(data['Long_MA'], label='200-day MA', alpha=0.7)

# Mark buy signals
plt.plot(data[data['Position'] == 1].index, 
         data['Short_MA'][data['Position'] == 1], 
         '^', markersize=10, color='g', lw=0, label='Buy Signal')

# Mark sell signals
plt.plot(data[data['Position'] == -1].index, 
         data['Short_MA'][data['Position'] == -1], 
         'v', markersize=10, color='r', lw=0, label='Sell Signal')

plt.title(f'{ticker} - Moving Average Crossover Strategy')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend(loc='best')
plt.show()
# Backtesting the strategy
initial_capital = 100000  # Initial capital in USD
positions = pd.DataFrame(index=data.index).fillna(0)

# Buy 100 shares on buy signal and sell 100 shares on sell signal
positions[ticker] = 100 * data['Signal'] 

# Calculate portfolio value
portfolio = positions.multiply(data['Close'], axis=0)
portfolio['Total'] = portfolio.sum(axis=1)

# Calculate returns
portfolio['Returns'] = portfolio['Total'].pct_change()

# Compare strategy returns to buy-and-hold returns
buy_and_hold_returns = data['Close'].pct_change().cumsum()

# Plot strategy performance
plt.figure(figsize=(12,8))
plt.plot(portfolio['Returns'].cumsum(), label='Strategy Returns', alpha=0.7)
plt.plot(buy_and_hold_returns, label='Buy-and-Hold Returns', alpha=0.7)
plt.title(f'{ticker} - Strategy vs. Buy-and-Hold')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend(loc='best')
plt.show()
