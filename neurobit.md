# Cryptocurrency Historical Data Extraction, Analysis, and Trading Bot

## Overview

This Python script is a comprehensive tool for cryptocurrency trading that integrates with the Coinbase Pro API. It allows users to extract historical cryptocurrency price data, perform technical analysis, visualize data, and implement automated trading strategies. The script also includes a Long Short-Term Memory (LSTM) neural network to predict future price movements.

## Features

1. **Historical Data Extraction**: 
   - Utilizes the `Coinbase Pro API` to gather historical data for a specified cryptocurrency ticker over a user-defined time period.
   - Supports different time granularities for data (e.g., 1 minute, 5 minutes, 1 hour, etc.).

2. **Technical Analysis and Indicators**:
   - Calculates the **Relative Strength Index (RSI)** and other technical indicators to assist in trading decisions.
   
3. **Data Visualization**:
   - Provides options to visualize historical price data and trading signals using libraries such as `matplotlib` and `plotly`.

4. **Automated Trading Bot**:
   - Implements buy and sell strategies based on market conditions.
   - Automatically places buy/sell orders on Coinbase Pro based on predefined criteria, including stop-loss limits and profit targets.
   
5. **LSTM Neural Network for Price Prediction**:
   - A deep learning model is built using an LSTM neural network to predict future price movements and assist in automated trading strategies.

## Prerequisites

- **Python** 3.7 or higher
- Libraries: 
  - `requests`, `pandas`, `numpy`, `matplotlib`, `librosa`, `tensorflow`, `sklearn`, `hmac`, `hashlib`, `http.client`, `json`, `time`, `base64`, `datetime`
  
You can install the required libraries using `pip`:

pip install requests pandas numpy matplotlib librosa tensorflow scikit-learn

Usage
The script provides the following main functions:

HistoricalData Class:

This class handles the data extraction from the Coinbase Pro API.
Example usage:
python
Copy code
data = HistoricalData('BTC-USD', 60, '2024-08-30-00-00').retrieve_data()
print(data)
deeper() Function:

This function handles the main trading logic, performing analysis, predictions, and trading actions.
It isolates features from the extracted data, applies technical indicators, and predicts future prices using LSTM.
Buy and Sell Logic:

The trading bot makes buying or selling decisions based on RSI, current price, and predicted values.
It communicates with the Coinbase API to execute trades.
Trading Strategy
Buying Strategy:

Buys when the current price is lower than the selling price, or when RSI is greater than or equal to 65.
Selling Strategy:

Sells when the current price is higher than the buying price, or when RSI is less than or equal to 40.
Stop-Loss Mechanism:

Implements stop-loss orders to minimize losses by triggering a sell if the price falls below a certain threshold.
LSTM Model for Price Prediction
The LSTM model predicts future prices based on historical data. The predictions help inform the trading strategy and can dynamically adjust buy/sell decisions.

Model Configuration:
Two LSTM layers with dense output.
Configurable hyperparameters such as batch size, time steps, and learning rate.
Training and Prediction:
The model is trained on historical low prices of the cryptocurrency, and future low prices are predicted.
Additional Notes
The script includes numerous helper functions and safety checks to ensure robust trading behavior.
Make sure to respect API rate limits to avoid getting blocked by the exchange.
Disclaimer: This script is intended for educational purposes only. Cryptocurrency trading involves significant risk, and you should carefully consider your objectives and risk appetite before trading.
