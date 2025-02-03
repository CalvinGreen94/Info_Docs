from flask import Flask, session, abort, request, jsonify, render_template, redirect, url_for, flash, redirect, Response
import os
# import wikipedia
import datetime
import hashlib
import json
from urllib.parse import urlparse
from flask_cors import CORS
import requests
import asyncio
from flask_bootstrap import Bootstrap
import openai
from flask_sslify import SSLify
import os
from sklearn.metrics import mean_squared_error
from ta.momentum import RSIIndicator

# from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
# from qiskit import IBMQ, Aer, transpile, assemble
# from qiskit.visualization import plot_histogram, plot_bloch_multivector, array_to_latex
# from qiskit.extensions import Initialize
# from qiskit.ignis.verification import marginal_counts
# from qiskit.quantum_info import random_statevector

# from qiskit import IBMQ
# IBMQ.save_account('')
# # IBMQ.disable_account()
# # IBMQ.disable_account()
# IBMQ.enable_account('')
# # IBMQ.backends()
# providers = IBMQ.providers()
# providers
# provider = IBMQ.get_provider(hub='ibm-q') 
# provider
# provider.backends()
# from qiskit.providers.ibmq import least_busy
# from qiskit.tools.monitor import job_monitor
import csv
import json
import time
import httpx
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from faunadb import query as q
from faunadb.client import FaunaClient
from faunadb.errors import BadRequest, NotFound
from solana.rpc.api import Client
import numpy as np
import datetime as dt
import time
from uuid import *

app = Flask(__name__)
CORS(app)
# app.config['BOOTSTRAP_BTN_STYLE'] = 'primary'  # default to 'secondary'
# app.config['BOOTSTRAP_BOOTSWATCH_THEME'] = 'lumen'
app.secret_key = str(uuid4()).replace('-', '')

fauna_ipa = '###'
helius_api_key = '###'
chainstack_url = '###'




EXPLORER_URL_ADD = "https://explorer.solana.com/address/"
EXPLORER_URL_TX = "https://explorer.solana.com/tx/"
ENDPOINT = f"https://solana-mainnet.core.chainstack.com/{chainstack_url}"
ADDRESS_RAYDIUM_AMM = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"
ADDRESS_LIQUIDITY_POOL = "5Q544fKrFoe6tsEbD7S8EmxGTJYAKtTVhAW5Q5pge4j1"
ADDRESS_SOLSCAN = "https://api.solscan.io/account?address="

# def save_data_to_json(data, filename='C:/Users/peace/desktop/hopeswap/autoswap/hope_bot/src/data.json'):
#     with open(filename, 'w') as json_file:
#         json.dump(data, json_file)
def get_next_filename(base_filename='data', extension='json'):
    i = 0
    while True:
        filename = f"{base_filename}_{i}.{extension}"
        if not os.path.exists(filename):
            return filename
        i += 1
def save_data_to_json(new_data, base_filename='data', extension='json'):
    filename = get_next_filename(base_filename, extension)
    
    # Save the new data to a new file
    with open(filename, 'w') as json_file:
        json.dump(new_data, json_file, indent=4)
    
    print(f"Data saved to {filename}")
    
    # # Read the existing data
    # with open(filename, 'r') as json_file:
    #     try:
    #         existing_data = json.load(json_file)
    #     except json.JSONDecodeError:
    #         existing_data = {}

    # # Update or add new data
    # if isinstance(existing_data, dict):
    #     existing_data.update(new_data)
    # else:
    #     print("Existing data is not a dictionary. Cannot update new data.")
    #     return

    # # Save the updated data
    # with open(filename, 'w') as json_file:
    #     json.dump(existing_data, json_file, indent=4)
client = FaunaClient(secret='###',domain="db.us.fauna.com")
import requests
import json

def get_token_accounts(token):
    all_owners = set()
    cursor = None
    url = f"https://mainnet.helius-rpc.com/?api-key={helius_api_key}"

    while True:
        params = {
            "limit": 1000,
            "mint": token
        }

        if cursor is not None:
            params["cursor"] = cursor

        body = {
            "jsonrpc": "2.0",
            "id": "helius-test",
            "method": "getTokenAccounts",
            "params": params
        }

        try:
            response = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(body))
            response.raise_for_status()  # Raise an exception for HTTP errors
        except requests.RequestException as e:
            print(f"Request failed: {e}")
            return {"error": str(e)}, 500

        data = response.json()

        if not data.get("result") or len(data["result"]["token_accounts"]) == 0:
            break

        for account in data["result"]["token_accounts"]:
            all_owners.add(account["owner"])

        cursor = data["result"].get("cursor")

    return list(all_owners)


root_node = '7yd5rX7Bpt6GmwqnVnstiPV22S268rLhDpMZtDhuzEoA'#New
def token_balance(wallet, token):
    url = f'https://mainnet.helius-rpc.com/?api-key={helius_api_key}'
    headers = {"accept": "application/json", "content-type": "application/json"}

    payload = {
        "id": 1,
        "jsonrpc": "2.0",
        "method": "getTokenAccountsByOwner",
        "params": [
            wallet,
            {"mint": token},
            {"encoding": "jsonParsed"},
        ],
    }
    try:
        response = requests.post(url, json=payload, headers=headers)
        response_json = response.json()
        # Check if the response contains the expected keys
        if "result" in response_json and "value" in response_json["result"]:
            amount = response_json["result"]["value"][0]["account"]["data"]["parsed"]["info"]["tokenAmount"]["uiAmount"]
            # print(amount)
            return amount
        else:
            print("Response format doesn't match expectations.")
            amount = 0
            return amount
    except Exception as e:
        print("An error occurred:", e)
        amount = 0
        return amount

helius_base_url = 'https://api.helius.xyz/v0'

# Replace with the public key of the account you want to check
public_key = root_node

# Construct the URL


def get_balance():
    # try:
    url = f'{helius_base_url}/addresses/{public_key}/balances?api-key={helius_api_key}'
    response = requests.get(url)
    response.raise_for_status()
    balance_data = response.json()
        
        # Print the raw JSON response for debugging
        # print(json.dumps(balance_data, indent=4))
        
        # Get the native balance in lamports
    lamports = balance_data.get('nativeBalance', 0)
        
        # Convert lamports to SOL (1 SOL = 1,000,000,000 lamports)
    sol = lamports / 1_000_000_000
        
        # Print balance with precision
    # print(f'Balance: {sol:.9f} SOL')
    balance = f'{sol:.9f}'
    # except requests.exceptions.RequestException as e:
    #     print(f'Error fetching balance: {e}')
    return balance
def sakujo(sakujo="NaN"):


    return
@app.route('/progress')
def progress():
    def generate():
        x = 0

        while x <= 100:
            yield "data:" + str(x) + "\n\n"
            x = x + 10
            time.sleep(0.5)

    return Response(generate(), mimetype='text/event-stream')
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/rpc', methods=['GET'])
def home():
    import time
    # time.sleep(10)

    while True:
        
        try:
            root_node = '7yd5rX7Bpt6GmwqnVnstiPV22S268rLhDpMZtDhuzEoA'#New

            import json

            # Path to your JSON file
            file_path = 'data/lp_v2.json'

            # Read and parse the JSON file
            try:
                
                with open(file_path, 'r') as file:
                    json_data = json.load(file)
            except Exception as e:
                print(f"Error reading or parsing JSON file: {e}")
                time.sleep(5)
                # exit(1)  # Exit if there's an error

            # Process data
            id_ = json_data['id']
            ownerBaseAta = json_data['ownerBaseAta']
            baseMint = json_data['baseMint']
            authority = json_data['authority']
            openOrders = json_data['openOrders']
            targetOrders = json_data['targetOrders']
            baseVault = json_data['baseVault']
            quoteVault = json_data['quoteVault']
            marketProgramId = json_data['marketProgramId']
            marketId = json_data['marketId']
            marketBids = json_data['marketBids']
            marketAsks = json_data['marketAsks']
            marketEventQueue = json_data['marketEventQueue']
            marketBaseVault = json_data['marketBaseVault']
            marketQuoteVault = json_data['marketQuoteVault']
            marketAuthority = json_data['marketAuthority']
            ownerQuoteAta = json_data['ownerQuoteAta']
            quoteMint = json_data['quoteMint']
            programId = json_data['programId']
            pool = json_data['pool']
            # isMintable = json_data['isMintable']
            # isMutable = json_data['isMutable']
            # frozen = json_data['isFrozen']
            mintA = json_data['baseMint']
            tokenA = json_data['quoteMint']
            precision = json_data['precision']
            isFrozen = json_data['isFrozen']
            # pools = json_data['pool']
            # pooled = json_data['pooled']
            
            

            # Output the results
            print(f'id: {id_}')
            # print(f'pool: {pooled}')
            print(f'quoteMint: {tokenA}')
            print(f'baseMint: {mintA}')
            print(f'precision: {precision}')
            print(f'isFrozen: {isFrozen}')

            # If needed to integrate with existing code (example)
            result = {
                'data': [json_data]  # Wrap json_data in a list for compatibility
            }

            # Extract precision if present
            precision_list = [json_data['precision']]
            if precision_list:
                for precision in precision_list:
                    precision = int(precision)  # Convert precision to int and print
                    print(f'precision {precision}')

            # Extract and print frozen status
            frozen = [json_data['isFrozen']]
            print(frozen)
            print(str(frozen))

            # if frozen is None:
            #     print('skipping')
            # mintable =[doc["data"]['isMintable'] for doc in result["data"]]
            # mutable =[doc["data"]['isMutable'] for doc in result["data"]]
            # print('frozen: {} \n mintable: {} \n mutable: {} \n decimals: {} \n'.format(frozen,mintable,mutable,decimals))
            # if frozen =='true' or mintable =='true' or mutable =='true' or decimals !='6':
            #     print('skipping')
            # print('MintA',mintA)
            # rsi = [doc["data"]['rsi'] for doc in result["data"]]
            # ammo = client.query(q.map_(q.lambda_("x", q.get(q.var("x"))), q.paginate(q.documents(q.collection("swapAmmo")))))
            # ammo_data = [doc["data"] for doc in result["data"]]
            # {'baseMint':base_address,
            # ammuntion = [doc["data"]['ammo'] for doc in result["data"]]

            # result_training = client.query(
            #             q.map_(
            #                 q.lambda_("x", q.get(q.var("x"))),
            #                 q.paginate(q.documents(q.collection("hopebot_datafeeds")))
            #             )
            #         )

            # predictions = client.query(
            #             q.map_(
            #                 q.lambda_("x", q.get(q.var("x"))),
            #                 q.paginate(q.documents(q.collection("hopebot_predictions")))
            #             )
            #         )


            if frozen=='true' or mintA=='So11111111111111111111111111111111111111112' :
                print('skipping, account can be frozen or base mint is solana quote ...')
                # Delete all rows in lp_v2 collection
                # for doc in result["data"]:
                #     client.query(q.delete(doc['ref']))
                # print('All lp_v2 rows deleted')

                # for doc in predictions["data"]:
                #     client.query(q.delete(doc['ref']))
                # print('All hopebot_predictions rows deleted')
                        
                # for doc in result_training["data"]:
                #     client.query(q.delete(doc['ref']))
                # print('All hopebot_datafeeds rows deleted')


            else:
                print('Frozen is not true, continuing')

            # if token_balance(root_node,to) ==0:
            #     print('skipping')
            if frozen!='true':
                # with open("deployed.txt", "w", encoding="utf-8", errors='ignore') as txt_file:
                #     for line in tokenA:
                #         txt_file.write("".join(line))
                with open("deployed.txt", "w", encoding="utf-8") as txt_file:
                    txt_file.write(tokenA)  # Write the entire tokenA string on a single line
                
                # Process data for each tokenA
                # time.sleep(2)

                for token in mintA:
                    # time.sleep(60)

                    # time.sleep(3)  # Pause for a while to avoid API rate limits
                    if frozen=='true'or mintA=='So11111111111111111111111111111111111111112' :
                        
                        print('skipping, account can be frozen or base mint is solana quote ...')
                    
                


                    else:
                        print('Frozen is not true, no rows deleted')
                    url = f"https://api.dexscreener.com/latest/dex/tokens/{tokenA}"
                    resp = httpx.get(url)
                    data = resp.json()["pairs"]
                    result = {'result': data}
                    print(url+'\n')
                    
                    
                    print(result)

                
                    time.sleep(2)
                    hopebot_datafeeds = 'hopebot_datafeeds.json'
                    if data:
                        try:
                            # Save data to JSON file
                            with open(hopebot_datafeeds, mode='w', encoding='utf-8') as file:
                                # Write JSON data to file
                                json.dump(result, file, indent=4)
                            
                            print(f"Data successfully saved to {hopebot_datafeeds}")
                        except Exception as e:
                            print(f"Error saving data to JSON: {e}")
                    else:
                        print('No data available to save.')
                    # Process data for training
                    csv_file_path = 'trainingv1.csv'

             
               
                    
                    with open(hopebot_datafeeds, 'r', encoding='utf-8') as file:
                        data = json.load(file)

                    result_training = data.get("result", [])

                    # If the items are nested under a "data" key, adjust as needed
                    data_training = [item for item in result_training]



                    # price_data = client.query(
                    #     q.map_(
                    #         q.lambda_("x", q.get(q.var("x"))),
                    #         q.paginate(q.documents(q.collection("prices")))
                    #     )
                    # )

                    # Extract the data
                    # pricing = [doc["data"] for doc in price_data["data"]]

                    # # Extract and convert sell and buy prices
                    # sol_sell_price_list = [doc['current_sell'] for doc in pricing]
                    # sol_buy_price_list = [doc['current_buy'] for doc in pricing]
                    # sol_true_price_list = [doc['solTradePrice (Open)'] for doc in pricing]


                    # # Get the last sell and buy prices and convert them to float
                    # sol_sell_price = float(sol_sell_price_list[-1])
                    # sol_buy_price = float(sol_buy_price_list[-1])
                    # sol_trade_price = float(sol_true_price_list[-1])



                    if frozen == 'true' or mintA=='So11111111111111111111111111111111111111112':
                        print('skipping, account can be frozen or base mint is solana quote ...')
               

                    else:
                        print('Frozen is not true, no rows deleted')

                    field_names = [
                        'Pair Address', 'Volume (24h)', 'Volume (1h)', 'Volume (5m)', 'Buys (5m)', 'Sells (5m)',
                        'Price (Native)', 'Liquidity (USD)', 'Price Change (5m)', 'amountIn', 'amountOut', 'Balance',
                        'solBalance', 'totalOut', 'burnAmount','transferAmount'
                    ] #'solPrice','solBuyPrice','solSellPrice','solTradePrice'

                    with open(csv_file_path, mode='a', newline='') as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=field_names)
                        for item in data_training:
        
                            pair_address = item.get('pairAddress', '')
                            base_address = item.get('baseToken', {}).get('address', '')
                            quote_address = item.get('quoteToken', {}).get('address', '')
                            h24_volume = item.get('volume', {}).get('h24', 0)
                            h1_volume = item.get('volume', {}).get('h1', 0)
                            m5_volume = item.get('volume', {}).get('m5', 0)
                            m5_buy = item.get('txns', {}).get('m5', {}).get('buys', 0)
                            m5_sell = item.get('txns', {}).get('m5', {}).get('sells', 0)
                            m5_change = item.get('priceChange', {}).get('m5', 0)
                            price = item.get('priceUsd', '')
                            liquidity = item.get('liquidity', {}).get('usd', 0)
                            liquiditySol = item.get('liquidity', {}).get('quote', 0)
                            dex = item.get('dexId', '')
                            balance = token_balance(root_node, tokenA)
                            amountIn = 10
                            amountIn = int(amountIn * 10000)
                            amountOut = balance 
                            burnAmount = amountOut // 4
                            
                            totalOut = int(amountOut) 
                            # transferAmount = float(balance)*float(price)
                            # transferAmount = int(transferAmount)
                            liquiditySol = float(liquiditySol)
                            # transferAmount = transferAmount
                            # transferAmount = int(transferAmount)

                            # print("Transfer Amount in SOL:", transferAmount)

                            if totalOut == "NaN":
                                totalOut = "0.00"
                            if burnAmount == "NaN":
                                burnAmount = "0.00"

                            amountOut = totalOut
                            solBalance = get_balance()
                            transferAmount = amountIn
                            # sol_sell_price = float(sol_sell_price)
                            # sol_buy_price = float(sol_buy_price)

                            if precision == 1:
                                totalOut = amountOut * 10   
                                burnAmount = burnAmount * 10                            
                            elif precision == 2:
                                totalOut = amountOut * 100  
                                burnAmount = burnAmount * 100    
                            elif precision == 3:
                                totalOut = amountOut * 1000 
                                burnAmount = burnAmount * 1000 
                            elif precision == 4:
                                totalOut = amountOut * 10000  
                                burnAmount = burnAmount * 10000    
                            elif precision == 5:
                                totalOut = amountOut * 100000
                                burnAmount = burnAmount * 100000   
                            elif precision == 6:
                                totalOut = amountOut * 1000000 
                                burnAmount = burnAmount * 1000000    
                            elif precision == 7:
                                totalOut = amountOut * 10000000  
                                burnAmount = burnAmount * 10000000   
                            elif precision == 8:
                                totalOut = amountOut * 100000000
                                burnAmount = burnAmount * 100000000     
                            elif precision == 9:
                                totalOut = amountOut * 1000000000  
                                burnAmount = burnAmount * 1000000000 

                            if price == "NaN":
                                price = "0.00"
                            # if sol_sell_price == "NaN":
                            #     sol_sell_price = 0.00
                            # if sol_buy_price == "NaN":
                            #     sol_buy_price = 0.00
                            # if sol_trade_price == "NaN":
                            #     sol_trade_price = 0.00
                            if amountIn == "NaN":
                                amountIn = "0"
                            if liquidity == "NaN":
                                liquidity = "0.00"
                            if liquiditySol == "NaN":
                                liquiditySol = "0.00"

                            if balance == "NaN":
                                balance = "0.00"
                            if transferAmount == "NaN":
                                transferAmount = "0"
                            # Write data to CSV
                            if dex != 'raydium':
                                print('Dex not raydium operable')
                                break
                                # pass
                            if dex == 'raydium':
                                writer.writerow({
                                    'Pair Address': pair_address,
                                    'Volume (24h)': h24_volume,
                                    'Volume (1h)': h1_volume,
                                    'Volume (5m)': m5_volume,
                                    'Buys (5m)': m5_buy,
                                    'Sells (5m)': m5_sell,
                                    'Price (Native)': price,
                                    'Liquidity (USD)': liquidity,
                                    'Price Change (5m)': m5_change,
                                    'amountIn': amountIn,
                                    'amountOut': amountOut,
                                    'Balance': balance,
                                    'solBalance': solBalance,
                                    'totalOut': totalOut,
                                    'burnAmount': burnAmount,
                                    'transferAmount': transferAmount
                                    # 'solPrice': liquiditySol,
                                    # 'solBuyPrice' : sol_buy_price,
                                    # 'solSellPrice' : sol_sell_price,
                                    # 'solTradePrice': sol_trade_price
                
                                })

                                
                                print('data written')
                                time.sleep(1)

                                print('performing linear regression')
                                # Preprocess and train model
                                os.makedirs('data', exist_ok=True)
                                os.makedirs('data/good/', exist_ok=True)
                                os.makedirs('data/mediocre/', exist_ok=True)
                                os.makedirs('data/bad/', exist_ok=True)

                                data = []

                                with open('trainingv1.csv', 'r') as file:
                                    reader = csv.reader(file)
                                    for row in reader:
                                        data.append(row)

                                pair_address_data = defaultdict(list)
                                for row in data:
                                    pair_address = row[0]
                                    pair_address_data[pair_address].append(row)

                                for pair_address, rows in pair_address_data.items():
                                    filename = 'data/' + pair_address + '.csv'
                                    with open(filename, 'w', newline='') as file:
                                        writer = csv.writer(file)
                                        writer.writerow(field_names)
                                        writer.writerows(rows)

                                    print('reg 4')
                                    reg4 = LinearRegression(n_jobs=-1)
                                    reg5 = LinearRegression(n_jobs=-1)
                                    reg6 = LinearRegression(n_jobs=-1)
                                    reg7 = LinearRegression(n_jobs=-1)
                                    reg8 = LinearRegression(n_jobs=-1)
                                    reg9 = LinearRegression(n_jobs=-1)
                                    reg10 = LinearRegression(n_jobs=-1)
                                    reg11 = LinearRegression(n_jobs=-1)


                                    # print('chart \n https://www.dexscreener.com/solana/{}'.format(pair_address))

                                    a = pd.read_csv(filename)  # Corrected to use the generated filename
                                    period = 14
                                    a['RSI (Native)'] = RSIIndicator(close=a['Price (Native)'], window=period).rsi()
                                    a['RSI (Native)'] = a['RSI (Native)'].fillna(0.0)
                                    a = a.drop(['Pair Address'], axis=1)
                                    # a = a.drop(['amountOut'], axis=1)
                                    # print(a)

                                    # b = pd.read_csv(filename) 
                                    # b.drop(['Price (Native)',axis=1])
                                    
                                    first_iteration = a
                                    second_iteration = a
                                    third_iteration = a 
                                    fourth_iteration = a
                                    fifth_iteration= a
                                    sixth_iteration= a
                                    seventh_iteration = a
                                    eigth_iteration = a 
                                    
                            
                                    y_volume = first_iteration['Liquidity (USD)']
                                    X_volume = first_iteration.drop(['Liquidity (USD)'], axis=1) 

                                    y_price = second_iteration['Price (Native)']
                                    X_price = second_iteration.drop(['Price (Native)'], axis=1)

                                    y_rsi = third_iteration['RSI (Native)']
                                    X_rsi = third_iteration.drop(['RSI (Native)'], axis=1)

                                    y_out = fourth_iteration['amountOut']
                                    X_out = fourth_iteration.drop(['amountOut'],axis=1)

                                    y_change = fifth_iteration['Price Change (5m)']
                                    X_change = fifth_iteration.drop(['Price Change (5m)'],axis=1)


                                    # y_buy = sixth_iteration['solBuyPrice']
                                    # X_buy = sixth_iteration.drop(['solBuyPrice'],axis=1)

                                    # y_sell = seventh_iteration['solSellPrice']
                                    # X_sell = seventh_iteration.drop(['solSellPrice'],axis=1)

                                    # y_open = eigth_iteration['solTradePrice']
                                    # X_open = eigth_iteration.drop(['solTradePrice'],axis=1)

                                    Xv_train, Xv_test, yv_train, yv_test = train_test_split(
                                        X_volume, y_volume, test_size=.45, shuffle=False)

                                    Xp_train, Xp_test, yp_train, yp_test = train_test_split(
                                        X_price, y_price, test_size=.45, shuffle=False)

                                    Xrsi_train, Xrsi_test, yrsi_train, yrsi_test = train_test_split(
                                        X_rsi, y_rsi, test_size=.22, shuffle=False)

                                    Xout_train, Xout_test, yout_train, yout_test = train_test_split(
                                        X_out, y_out, test_size=.22, shuffle=False)

                                    X_change_train, X_change_test, y_change_train, y_change_test = train_test_split(
                                        X_change, y_change, test_size=.32, shuffle=False)
                                    
                                    # X_buy_train, X_buy_test, y_buy_train, y_buy_test = train_test_split(
                                    #     X_buy, y_buy, test_size=.32, shuffle=False)
                                    
                                    # X_sell_train, X_sell_test, y_sell_train, y_sell_test = train_test_split(
                                    #     X_sell, y_sell, test_size=.32, shuffle=False)
                                    # X_open_train, X_open_test, y_open_train, y_open_test = train_test_split(
                                    #     X_open, y_open, test_size=.32, shuffle=False)
                                    
                                    mini = MinMaxScaler()
                                    #transform / scale model
                                    X_volume = mini.fit_transform(X_volume)
                                    X_price = mini.fit_transform(X_price)
                                    X_rsi = mini.fit_transform(X_rsi)
                                    X_out = mini.fit_transform(X_out)
                                    X_change = mini.fit_transform(X_change)
                                    # X_buy = mini.fit_transform(X_buy)
                                    # X_sell = mini.fit_transform(X_sell)
                                    # X_open = mini.fit_transform(X_open)
                                    
                                    #fit model
                                    reg4.fit(Xv_train, yv_train)
                                    score = reg4.score(Xv_test, yv_test)
                                    pred = reg4.predict(Xv_test[-1:])

                                    reg5.fit(Xp_train, yp_train)
                                    score1 = reg5.score(Xp_test, yp_test)
                                    pred1 = reg5.predict(Xp_test[-1:])


                                    reg6.fit(Xrsi_train, yrsi_train)
                                    score2 = reg6.score(Xrsi_test, yrsi_test)
                                    pred2 = reg6.predict(Xrsi_test[-1:])
                                    true_rsi = a['RSI (Native)']
                                    rsi_last = float(true_rsi[-1:])

                                    print('RSI LAST:',rsi_last)
                                    # print('{}'.format(float(balance)*float(price)))

                                    # reg7.fit(float(Xout_train), float(yout_train))
                                    # score7 = reg7.score(Xout_test, yout_test)
                                    # pred_out = reg7.predict(Xout_test[-1:])
                                    # pred_out = int(pred_out)
                                    # # print('RSI LAST:',rsi_last)
                                    # print('Amount Out Prediction: {}'.format(pred_out))
                                    
                                    reg8.fit(X_change_train, y_change_train)
                                    score8 = reg8.score(X_change_test, y_change_test)
                                    pred_price_change = reg8.predict(X_change_test[-1:])
                                    pred_price_change = float(pred_price_change)
                                    print('predicted price change {}'.format(pred_price_change))


                                    # reg9.fit(X_sell_train, y_sell_train)
                                    # score9 = reg9.score(X_sell_test, y_sell_test)
                                    # pred_sell = reg9.predict(X_sell_test[-1:])
                                    # pred_sell = float(pred_sell)
                                    # print('predicted sell price {}'.format(pred_sell))
                                    # print(f'\n\n score {score9}')

                                    # reg10.fit(X_buy_train, y_buy_train)
                                    # score10 = reg10.score(X_buy_test, y_buy_test)
                                    # pred_buy = reg10.predict(X_buy_test[-1:])
                                    # pred_buy = float(pred_buy)
                                    # print('predicted buy price {}'.format(pred_buy))
                                    # print(f'\n\n score {score10}')

                                    # solana_price = a['solTradePrice'][-1:]
                                    # solana_price = float(solana_price)
                                    # print(f'SOLANA PRICE: {solana_price}')

                                    # solana_buy = a['solBuyPrice'][-1:]
                                    # solana_buy = float(solana_buy)
                                    # print(f'SOLANA Buy PRICE: {solana_buy}')

                                    # solana_sell = a['solSellPrice'][-1:]
                                    # solana_sell = float(solana_sell)
                                    # print(f'SOLANA Sell PRICE: {solana_sell}')


                                    # reg11.fit(X_open_train, y_open_train)
                                    # score11 = reg11.score(X_open_test, y_open_test)
                                    # pred_open = reg11.predict(X_open_test[-1:])
                                    # pred_open = float(pred_open)
                                    # print('predicted open price {}'.format(pred_open))
                                    # print(f'\n\n score {score11}')




                                    print('{} Score: {}, Price Change Prediction: {}, \n with a predicted RSI of {}, \n having a score of {} '.format(pair_address,score1,pred1,pred2,score2))
            # price =
                                    amountOut=a['amountOut'][-1:]
                                    amountOut = int(amountOut)
                                    amountOut = amountOut
                                    print('OOUUTTTT',amountOut) 
                                    solBalance = a['solBalance'][-1:]
                                    print(f'SOL balance{solBalance}') 
                                    solBalance = float(solBalance)
                                    rssi = a['RSI (Native)'][-1:]
                                    rssi = float(rssi)
                                    current_price = float(price[-1:])
                                    print(f'current price: {current_price}')
                                    price2 = a['Price (Native)'][-1:]
                                    price2 = float(price2)
                                    print(f'price2: {price2}')
                                    print(f'predicted price: {pred1}')
                                    balance = a['Balance'][-1:]
                                    balance = int(balance) 
                                    slippage_tolerance = 0.027
                                    # price3 = a['Price (Native)'][-1:]
                                    # price3= float(price3)
                                    effective_price = float(price) * (1 + slippage_tolerance)
                                    priceChange = a['Price Change (5m)']

                                    if float(a['Price (Native)'][-1:]) == current_price or float(pred1) == current_price:
                                        print('HOLDING DOING NOTHING')


                                                                                                #or pred_buy<=pred_sell or pred_open<pred_sell    
                                    if (float(pred1) <= effective_price or float(priceChange) <0.04 or  pred_price_change < 0.0 or (float(first_iteration['Liquidity (USD)'][-1:]) >= 3550.00 and solBalance >= 0.01) or rsi_last >= 20.00  or float(price)<float(a['Price (Native)'][-1:])) and float(pred) >= 50.00 or amountOut<=0:

                                    # if float(pred1) <= float(price) or float(first_iteration['Liquidity (USD)'][-1:]) >=3.00 and solBalance>=0.029 or rssi>=20.00: # or float(pred2)>=42.0 and float(pred2)<=93

                                        # if frozen =='false' and float(pred) >=1000.00 or float(first_iteration['Liquidity (USD)'][-1:]) >=2.00 and solBalance>=0.029 or rssi>=20.00:
                                        print("account can't be frozen, continuing to BUY")
                                        side = 'Buy'
                                        balance = a['Balance'][-1:]
                                        balance = int(balance) 
                                            # if side == 'Buy' and balance<0 or side == 'Buy' and balance>0:

                                            # amount = 1000000000
                                        message = '{} Predicted Price Change is {}. Filtering Bad Data'.format(pair_address,pred)
                                        pred1_int = int(pred1)
                                        pred_int = int(pred)
                                        root_node = '7yd5rX7Bpt6GmwqnVnstiPV22S268rLhDpMZtDhuzEoA'
                                        amountIn =a['amountIn'][-1:]
                                        amountIn = int(amountIn)
                                        print(amountIn)
                                        transferAmount = a['transferAmount'][-1:]
                                        transferAmount = int(transferAmount)
                                        # a['side'] = 0
                                        # # side = a['side']
                                        # print(a)
                                        # a.to_csv('side.csv')
                                        holders = get_token_accounts('UQ5q7mjDRynZnV5RX5YfbaJMnXdmRaSX61c2J2wBsoi')
                                        # token_holders = get_token_accounts(base_address)
                                        execute = {
                                            "id": id_,
                                            "decimals": precision,
                                            "authority": authority,
                                            "openOrders": openOrders,
                                            "targetOrders": targetOrders,
                                            "baseVault": baseVault,
                                            "quoteVault": quoteVault,
                                            "marketProgramId": marketProgramId,
                                            "marketId": marketId,
                                            "marketBids": marketBids,
                                            "marketAsks": marketAsks,
                                            "marketEventQueue": marketEventQueue,
                                            "marketBaseVault": marketBaseVault,
                                            "marketQuoteVault": marketQuoteVault,
                                            "marketAuthority": marketAuthority,
                                            "ownerQuoteAta": ownerQuoteAta,
                                            "ownerBaseAta": ownerBaseAta,
                                            "quoteMint": quoteMint,
                                            "baseMint": baseMint,
                                            "programId": programId,
                                            "pool": pool,
                                            "isFrozen": isFrozen,
                                            # "isMintable": isMintable,
                                            # "isMutable": isMutable,
                                            "precision": precision,
                                            'baseMint': base_address,
                                            'quoteMint': quote_address,
                                            'pair_address': pair_address,
                                            'price_score': float(score1),
                                            'rsi_score': float(score2),
                                            'liquidity_prediction': float(pred),
                                            'rating': '3',
                                            'side': side,
                                            'price': float(price),
                                            'price_prediction': float(pred1),
                                            'amount': amountIn,
                                            'rsi_prediction': float(pred2),
                                            'Balance': balance,
                                            'effective_price':effective_price,
                                            'hopeHolders':holders,
                                            'balances': {},
                                            # 'trading_balance': {}
                                        }
                                        # Save the data
                                        # if isinstance(holders, list):
                                        #     # Create a dictionary to store holder addresses and their respective balances
                                        #     all_balances = {}
                                        #     for holder in holders:
                                        #         balances = token_balance(holder, 'UQ5q7mjDRynZnV5RX5YfbaJMnXdmRaSX61c2J2wBsoi')
                                        #         print(f'DEBUG: holder={holder}, balance={balances}')
                                        #         # Add holder and their balance to the all_balances dictionary
                                        #         all_balances[holder] = balances
                                            
                                        #     # Add all balances to the execute['balances']
                                        #     execute['balances'] = all_balances

                                        # # if isinstance(token_holders, list):
                                        # #     # Create a dictionary to store holder addresses and their respective balances
                                        # #     mint_balance = {}
                                        # #     for token_holder in token_holders:
                                        # #         token_balances = token_balance(token_holder, base_address)
                                        # #         print(f'DEBUG: holder={token_holder}, balance={balances}')
                                        # #         # Add holder and their balance to the all_balances dictionary
                                        # #         mint_balance[token_holder] = token_balances

                                        # #     execute['trading_balance'] = mint_balance
                                        # else:
                                        #     print("'holders' is not iterable")
                                            

                                        # Save the data
                                        # time.sleep(1)
                                        print('saving execution data')
                                        save_data_to_json(execute)


                                        post = {'baseMint':base_address,'quoteMint':quote_address,'pair_address':pair_address,'price_score':float(score1),'rsi_score':float(score2),'liquidity_prediction':float(pred),'rating':'3','side':side,'price':float(price),'price_prediction':float(pred1),'amount':amountIn,'rsi_prediction':float(pred2),'Balance':balance,'effective_price':effective_price,'precision':precision,'transferAmount':transferAmount} #'predictedSellPrice':pred_sell,'predictedBuyPrice':pred_buy,'buyPrice':solana_buy,'sellPrice':solana_sell,'solanaPrice':solana_price,'predictedSolanaPrice':pred_open}
                                            # time.sleep(2)
                                        # client.query(q.create(
                                        #             q.collection('hopebot_predictions'),
                                        #             {'data':post}
                                        #         ))
                                    
                                        print('{} Predicted Price Change is {}. Filtering Good Data'.format(pair_address,float(pred1)))
                        
                                        print(f'Buy Operation Data: {execute}')
                                        
                                    amountIn = amountIn
                                    price3 = a['Price (Native)'][-2:-1]
                                    price3 = float(price3)
                                    target_sell_price =price3 * 100.0000003  # 3% higher than the base price
                                    current_price = float(price)
                                    print(f'target sell price: {target_sell_price}')
                                    print(f'current price: {current_price}')
                                    price2 = a['Price (Native)']
                                    # price2 = float(price2)
                                    print(f'price2: {price2}')
                                    print(f'predicted price: {pred1}')
                                    balance = a['Balance'][-1:]
                                    balance = int(balance)
                                    print('balance: ',balance)

                                    priceChange = a['Price Change (5m)']
                                    # priceChange = float(priceChange

                                    effective_price = float(price) * (1 - slippage_tolerance)
                                                        #or pred_sell>=pred_buy   #or pred_sell>pred_open
                                    if (solBalance <= 0.01  or rsi_last <= 15.00 or float(priceChange) >=100.00 or float(priceChange) >=.97 or pred_price_change >= 0.80 or float(effective_price) >= float(pred1)  or float(price) >=float(pred1)  or float(price2[-1:])>=target_sell_price or float(priceChange[-1:])>=100.00) and frozen == 'false' and balance >= amountOut or balance >= amountOut and float(first_iteration['Liquidity (USD)'][-1:]) >= 3565.00 or  balance >= amountOut and float(first_iteration['Liquidity (USD)'][-1:]) <= 3400.00: #or float(pred2) >=53.0
                                        #^float(pred1) >= float(price)^
                                        balance = a['Balance'][-1:]
                                        balance = int(balance)
                                        print('balance: ',balance)
                                        if balance <= 0:
                                            print('skipping due to balance <= 0')
                                            continue

                                        # elif float(price) <target_sell_price:
                                        #     print('skipping due to target sell price not hit <= 0')
                                        #     continue

                                        elif amountOut > balance:
                                            print('amount out greater than balance, skipping')
                                            continue
                                        elif amountOut == 0:
                                            print('amount out is zero, skipping')
                                            continue
                                        elif frozen == 'true':
                                            print('skipping, account can be frozen')

                                            
                                        # if (float(pred1) >= float(price) or solBalance <= 0.028 or rsi_last <= 15.00) and frozen == 'false' and balance >= amountOut and float(first_iteration['Liquidity (USD)'][-1:]) <= 1.00: #float(pred) <=400.00 or balance >=amountOut and float(first_iteration['Liquidity (USD)'][-1:]) >=8000.00 or balance >=amountOut and float(first_iteration['Liquidity (USD)'][-1:]) <=float(first_iteration['Liquidity (USD)'][-2:-1:])  or balance >=amountOut and
                                            # time.sleep(2)
                                        rating = '1'
                                        side = 'Sell'
                                        print("account can't be frozen, continuing to Sell")
                                        print(amountOut)

                                        message = '{} Predicted Price Change is {}. Filtering Bad Data'.format(pair_address,pred)
                                        pred1_int = int(pred1)
                                        pred_int = int(pred)
                                            # amountIn = amountIn
                                            # amountOut = amountIn*float(price)
                                            # amountOut = balance_loader(amountOut)

                                        amountOut=a['totalOut'][-1:]
                                        amountOut = int(amountOut)
                                        transferAmount = a['transferAmount'][-1:]
                                        transferAmount = int(transferAmount)
                                            # Perform the multiplication
                                            # amountOut = balance_loader(base_address,root_node)
                                        holders = get_token_accounts('UQ5q7mjDRynZnV5RX5YfbaJMnXdmRaSX61c2J2wBsoi')
                                        # token_holders = get_token_accounts(base_address)

                                        execute = {
                                            "id": id_,
                                            "decimals": precision,
                                            "authority": authority,
                                            "openOrders": openOrders,
                                            "targetOrders": targetOrders,
                                            "baseVault": baseVault,
                                            "quoteVault": quoteVault,
                                            "marketProgramId": marketProgramId,
                                            "marketId": marketId,
                                            "marketBids": marketBids,
                                            "marketAsks": marketAsks,
                                            "marketEventQueue": marketEventQueue,
                                            "marketBaseVault": marketBaseVault,
                                            "marketQuoteVault": marketQuoteVault,
                                            "marketAuthority": marketAuthority,
                                            "ownerQuoteAta": ownerQuoteAta,
                                            "ownerBaseAta": ownerBaseAta,
                                            "quoteMint": quoteMint,
                                            "baseMint": baseMint,
                                            "programId": programId,
                                            "pool": pool,
                                            "isFrozen": isFrozen,
                                            # "isMintable": isMintable,
                                            # "isMutable": isMutable,
                                            "precision": precision,                                
                                            'baseMint': base_address,
                                            'quoteMint': quote_address,
                                            'pair_address': pair_address,
                                            'price_score': float(score1),
                                            'rsi_score': float(score2),
                                            'liquidity_prediction': float(pred),
                                            'rating': '1',
                                            'side': side,
                                            'price': float(price),
                                            'price_prediction': float(pred1),
                                            'amount': amountOut,
                                            'rsi_prediction': float(pred2),
                                            'Balance': balance,
                                            'effective_price':effective_price,
                                            'hopeHolders':holders,
                                            'balances': {},
                                            # 'trading_balance': {}
                                        }
                                        # Save the data
                                        # if isinstance(holders, list):
                                        #     # Create a dictionary to store holder addresses and their respective balances
                                        #     all_balances = {}
                                        #     for holder in holders:
                                        #         balances = token_balance(holder, 'UQ5q7mjDRynZnV5RX5YfbaJMnXdmRaSX61c2J2wBsoi')
                                        #         print(f'DEBUG: holder={holder}, balance={balances}')
                                        #         # Add holder and their balance to the all_balances dictionary
                                        #         all_balances[holder] = balances
                                            
                                        #     # Add all balances to the execute['balances']
                                        #     execute['balances'] = all_balances

                                        # # if isinstance(token_holders, list):
                                        # #     # Create a dictionary to store holder addresses and their respective balances
                                        # #     mint_balance = {}
                                        # #     for token_holder in token_holders:
                                        # #         token_balances = token_balance(token_holder, base_address)
                                        # #         print(f'DEBUG: holder={token_holder}, balance={balances}')
                                        # #         # Add holder and their balance to the all_balances dictionary
                                        # #         mint_balance[token_holder] = token_balances

                                        # #     execute['trading_balance'] = mint_balance
                                        # else:
                                        #     print("'holders' is not iterable'")
                                            
                                        # time.sleep(1)
                                        save_data_to_json(execute)
                                        print('SAVING EXECUTION DATA')
                                        node_address = pair_address
                                        root_node = '7yd5rX7Bpt6GmwqnVnstiPV22S268rLhDpMZtDhuzEoA'
                                        post = {'baseMint':base_address,'quoteMint':quote_address,'pair_address':pair_address,'price_score':float(score1),'rsi_score':float(score2),'liquidity_prediction':float(pred),'rating':'1','side':'Sell','price':float(price),'price_prediction':float(pred1),'amount':amountOut,'rsi_prediction':float(pred2),'Balance':balance,'effective_price':effective_price,'precision':precision,'transferAmount':transferAmount}# 'predictedSellPrice':pred_sell,'predictedBuyPrice':pred_buy,'buyPrice':solana_buy,'sellPrice':solana_sell,'solanaPrice':solana_price,'predictedSolanaPrice':pred_open
                                        # client.query(q.create(
                                        #         q.collection('hopebot_predictions'),
                                        #         {'data':post}
                                        #     ))
                                        print(f'Sell Operation Data: {execute}')
                     
                    # return jsonify({'status': 'success',  'message': 'Operation completed', 'data': execute}), 200
                    

        except Exception as e:
            print(f"Error occurred: {e}")
            # return jsonify({"status": "error", "message": str(e)}), 500

# if __name__ == "__main__":
    
#     app.run(debug=True,host="0.0.0.0",port=6009)

home()
