OverView
---

## NukeBot - LSTM Model

NukeBot utilizes a Long Short-Term Memory (LSTM) model for real-time market prediction and trading on the Solana blockchain.

### Model Parameters

- **LSTM Configuration**:
  - **Layers**: 2 LSTM layers with 5 and 1 units respectively.
  - **Input Shape**: Variable sequence length (defined by `time_steps`).
  - **Optimizer**: Adam optimizer with default parameters.
  - **Loss Function**: Mean Squared Error (MSE).
  - **Learning Rate**: 0.018.
  
- **Execution Logic**:
  - **Stop-loss Trigger**: If `current_price` falls below `stop_loss_price`, executes a sell order.
  - **Sell Order**: Places a sell order based on predicted price (`yh[-1]`).
  - **API Integration**: Uses Coinbase API for order placement and authentication.

### Usage

- **Training**: Trains on historical data split into training and testing sets.
- **Prediction**: Predicts future prices using LSTM model.
- **Execution**: Executes buy or sell orders based on model predictions and market conditions.

---

## KirbySwap - Linear Regression Model

KirbySwap implements Linear Regression for analyzing liquidity and price trends on the Solana blockchain.

### Model Parameters

- **Regression Setup**:
  - **Core Model**: Linear Regression model.
  - **Real-time Insights**: Triggers buy orders if `current_price` is below `current_sell` or if RSI >= 65.00.
  - **Data Scaling**: Uses MinMaxScaler for data normalization.

- **API Integration**:
  - **Order Placement**: Executes buy orders based on predictions.
  - **Authentication**: HMAC-based authentication with Coinbase API.

### Usage

- **Predictive Analysis**: Uses historical data to predict future price movements.
- **Trade Execution**: Places buy orders based on model predictions and market signals.
- **Iterative Process**: Continuously runs for a set number of iterations with a sleep interval between each iteration.
