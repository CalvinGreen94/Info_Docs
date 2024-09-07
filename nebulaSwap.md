# NebulaSwap Transaction Script

This script performs a series of operations to handle swap transactions using Solana. The primary functions involve reading JSON data, preparing transactions, and executing swaps on the Solana blockchain. Below is a detailed description of the script's functionality and usage.

## Overview

The script continuously processes swap transactions based on data read from a JSON file. It handles different aspects of token swaps including account creation, transaction preparation, and execution.

## Script Functions

### `swapTx`

The `swapTx` function manages the swap transactions by performing the following steps:

1. **Read JSON Data**: Continuously reads and parses JSON data from the specified `filePath`.

2. **Data Extraction**: Extracts `PredictionData` and `LiquidityData` from the JSON. These data sets are used to determine the details of the swap transactions.

3. **Account Processing**:
    - **Check for Frozen Accounts**: Skips processing for accounts marked as frozen.
    - **Process Prediction Data**: Utilizes prediction data for making decisions about the swaps.

4. **Transaction Preparation**:
    - **Create New Mint**: Function to create a new mint for tokens.
    - **Check Mint Authority**: Functions to verify if a token has mint or freeze authority.
    - **Create Transactions**: Constructs the necessary instructions and transactions for the swap operation.

5. **Execute Transactions**:
    - **Handle Buying and Selling**: Based on the `side` (Buy or Sell), the script adds appropriate instructions to the transaction.
    - **Send Transaction**: Submits the transaction to the Solana network and confirms its execution.
