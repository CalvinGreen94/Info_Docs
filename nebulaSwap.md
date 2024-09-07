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
  

Example data:

Here is your data formatted in Markdown:

```
## Data Object

```datastream example
{
    "id": "5YZMMrcKBecsKom4xSmsJhG4yWXLjz76Ncu2AA9qsM9Z",
    "decimals": 6,
    "authority": "5Q544fKrFoe6tsEbD7S8EmxGTJYAKtTVhAW5Q5pge4j1",
    "openOrders": "HZvHrdeBX287cyfE7PniTPoHc4HF3tvWcSHZpyuNV1yg",
    "targetOrders": "AJafo8PZcKrqJKxQQgdnpRGrYP78RyLcsJJguhqTtbx7",
    "baseVault": "CPfMChH3mTMRRuWS2eSfptrfXEc5ek4f1Qj8mbMtyqML",
    "quoteVault": "7qMD3hyQrEjRcdXj5QkzhQM9Cv5Mm1ybXDT3EqPWkG3v",
    "marketProgramId": "srmqPvymJeFKQ4zGQed1GFppgkRHL9kaELCbyksJtPX",
    "marketId": "G53db3in179VurEGDdWaHNz9Xdh7svQWsfDYXLHtDdGu",
    "marketBids": "4E9fmCD4vDtCc5F2e9dDmpxRWZijZzGbrhEerkKzXMs6",
    "marketAsks": "4JjyLhUjfNRyiH8ezDtk3HJVkDYsjWVrTuGgmS4kVVPr",
    "marketEventQueue": "FC1ESSG7Y6uzuGdvXwbCibEvGxLN8jdH88msMGbYrN7g",
    "marketBaseVault": "CGxyhEb2mAH8rVWypPDMG7DuS61Up3vuKqg49fyJ13kt",
    "marketQuoteVault": "GMphnyjFGwTQVVwXoUkx7QMNNYn7FrcsQJPP81cAtuqg",
    "marketAuthority": "7CxtFLjWCnvS1mdCwL4QNaq4sHyTX8aqUWMxBYkgVmW9",
    "ownerQuoteAta": "7AHiJeux4nv7SoVGWM62ZSzYzJ1omDPuZP9eme6xaeZs",
    "ownerBaseAta": "B8LyqZUK3uptmeCV8ZJrfWg9Yd4a6gTEt4LxR5XXHdrs",
    "quoteMint": "So11111111111111111111111111111111111111112",
    "baseMint": "2MRkhw8eTKKa2fgRED9AhZxQVZN54eymrMvbkrdrqgpf",
    "programId": "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8",
    "pool": "2MRkhw8eTKKa2fgRED9AhZxQVZN54eymrMvbkrdrqgpf",
    "isFrozen": "false",
    "isMintable": "false",
    "isMutable": "false",
    "precision": 6,
    "pair_address": "5YZMMrcKBecsKom4xSmsJhG4yWXLjz76Ncu2AA9qsM9Z",
    "price_score": 1.0,
    "rsi_score": 1.0,
    "liquidity_prediction": 1.6500000000000001,
    "rating": "3",
    "side": "Buy",
    "price": 2.083e-08,
    "price_prediction": 2.083e-08,
    "amount": 1000000,
    "rsi_prediction": 0.0,
    "Balance": 0,
    "effective_price": 2.1392409999999996e-08
}
```
