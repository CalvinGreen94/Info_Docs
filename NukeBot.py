import requests
import json
import time
from random import randint
import sys
import pandas as pd
import requests
from typing import *
import librosa
import json
import time
import pandas as pd
import sys
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import pickle
import numpy as np
import jsonify
from sklearn.neural_network import MLPRegressor
import keras
import os
os.environ['KERAS_BACKEND' ] = 'tensorflow'
os.environ['MKL_THREADING_LAYER'] = 'GNU'
from sklearn.neural_network import MLPRegressor
import math
from keras.models import Sequential
from keras.layers import Dense,LSTM
from keras.callbacks import TensorBoard
import keras
from keras.optimizers import SGD
import pandas as pd
import numpy as np
from keras import optimizers
# from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, Dense
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout, Activation
import datetime
from ta.momentum import RSIIndicator
# from tensorflow.keras.layers.merge import concatenate
from faunadb import query as q
from faunadb.client import FaunaClient
from faunadb.errors import BadRequest, NotFound
# from keras.models import load_model
# import tensorflow as tf 
# # from tensorflow.keras.models import model_from_json
# from keras.models import save_model, load_model
import matplotlib.pyplot as plt
import math
import os
# os.environ['KERAS_BACKEND' ] = 'tensorflow'
# os.environ['MKL_THREADING_LAYER'] = 'GNU'
# from keras.models import Sequential
# from keras.layers import Dense,LSTM
# # from keras.callbacks import TensorBoard
# import keras
# from keras.optimizers import SGD
import numpy as np
# from keras import optimizers
# from keras.models import Model
# from keras.layers import Input, Dense
# from keras.layers.normalization import BatchNormalization
# from keras.layers.core import Dropout, Activation
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import os
from web3.middleware import geth_poa_middleware
from web3.gas_strategies.time_based import medium_gas_price_strategy
from eth_account.messages import encode_defunct
from web3 import Web3
from coinbase.wallet.client import Client
import json
import pandas as pd 
import warnings
import numpy as np
import hashlib
import hmac
import time
import http.client
ipa = '###'
api_key = '###'
api_secret = '###'
fauna_ipa = '###'
import json
#infura_url = f"https://goerli.infura.io/v3/{ipa}"
#web3 = Web3(Web3.HTTPProvider(infura_url))
#web3.middleware_onion.inject(geth_poa_middleware, layer=0)
def save_data_to_json(data, filename='C:/Users/peace/desktop/hopeswap/hope_bot/src/data.json'):
    with open(filename, 'w') as json_file:
        json.dump(data, json_file)





Fclient = FaunaClient(secret=f'{fauna_ipa}',domain="db.us.fauna.com")

class HistoricalData(object):
    """
    This class provides methods for gathering historical price data of a specified
    Cryptocurrency between user specified time periods. The class utilises the CoinBase Pro
    API to extract historical data, providing a performant method of data extraction.
    
    Please Note that Historical Rate Data may be incomplete as data is not published when no 
    ticks are available (Coinbase Pro API Documentation).

    :param: ticker: a singular Cryptocurrency ticker. (str)
    :param: granularity: the price data frequency in seconds, one of: 60, 300, 900, 3600, 21600, 86400. (int)
    :param: start_date: a date string in the format YYYY-MM-DD-HH-MM. (str)
    :param: end_date: a date string in the format YYYY-MM-DD-HH-MM,  Default=Now. (str)
    :param: verbose: printing during extraction, Default=True. (bool)
    :returns: data: a Pandas DataFrame which contains requested cryptocurrency data. (pd.DataFrame)
    """
    def __init__(self,
                 ticker,
                 granularity,
                 start_date,
                 end_date=None,
                 verbose=True):

        if verbose:
            print("Checking input parameters are in the correct format.")
        if not all(isinstance(v, str) for v in [ticker, start_date]):
            raise TypeError("The 'ticker' and 'start_date' arguments must be strings or None types.")
        if not isinstance(end_date, (str, type(None))):
            raise TypeError("The 'end_date' argument must be a string or None type.")
        if not isinstance(verbose, bool):
            raise TypeError("The 'verbose' argument must be a boolean.")
        if isinstance(granularity, int) is False:
            raise TypeError("'granularity' must be an integer object.")
        if granularity not in [60, 300, 900, 3600, 21600, 86400]:
            raise ValueError("'granularity' argument must be one of 60, 300, 900, 3600, 21600, 86400 seconds.")

        if not end_date:
            end_date = datetime.today().strftime("%Y-%m-%d-%H-%M")
        else:
            end_date_datetime = datetime.strptime(end_date, '%Y-%m-%d-%H-%M')
            start_date_datetime = datetime.strptime(start_date, '%Y-%m-%d-%H-%M')
            while start_date_datetime >= end_date_datetime:
                raise ValueError("'end_date' argument cannot occur prior to the start_date argument.")

        self.ticker = ticker
        self.granularity = granularity
        self.start_date = start_date
        self.start_date_string = None
        self.end_date = end_date
        self.end_date_string = None
        self.verbose = verbose

    def _ticker_checker(self):
        """This helper function checks if the ticker is available on the CoinBase Pro API."""
        if self.verbose:
            print("Checking if user supplied is available on the CoinBase Pro API.")

        tkr_response = requests.get("https://api.exchange.coinbase.com/products")
        if tkr_response.status_code in [200, 201, 202, 203, 204]:
            if self.verbose:
                print('Connected to the CoinBase Pro API.')
            response_data = pd.json_normalize(json.loads(tkr_response.text))
            ticker_list = response_data["id"].tolist()

        elif tkr_response.status_code in [400, 401, 404]:
            if self.verbose:
                print("Status Code: {}, malformed request to the CoinBase Pro API.".format(tkr_response.status_code))
            sys.exit()
        elif tkr_response.status_code in [403, 500, 501]:
            if self.verbose:
                print("Status Code: {}, could not connect to the CoinBase Pro API.".format(tkr_response.status_code))
            sys.exit()
        else:
            if self.verbose:
                print("Status Code: {}, error in connecting to the CoinBase Pro API.".format(tkr_response.status_code))
            sys.exit()

        if self.ticker in ticker_list:
            if self.verbose:
                print("Ticker '{}' found at the CoinBase Pro API, continuing to extraction.".format(self.ticker))
        else:
            raise ValueError("""Ticker: '{}' not available through CoinBase Pro API. Please use the Cryptocurrencies 
            class to identify the correct ticker.""".format(self.ticker))

    def _date_cleaner(self, date_time: (datetime, str)):
        """This helper function presents the input as a datetime in the API required format."""
        if not isinstance(date_time, (datetime, str)):
            raise TypeError("The 'date_time' argument must be a datetime type.")
        if isinstance(date_time, str):
            output_date = datetime.strptime(date_time, '%Y-%m-%d-%H-%M').isoformat()
        else:
            output_date = date_time.strftime("%Y-%m-%d, %H:%M:%S")
            output_date = output_date[:10] + 'T' + output_date[12:]
        return output_date

    def retrieve_data(self):
        """This function returns the data."""
        if self.verbose:
            print("Formatting Dates.")

        self._ticker_checker()
        self.start_date_string = self._date_cleaner(self.start_date)
        self.end_date_string = self._date_cleaner(self.end_date)
        start = datetime.strptime(self.start_date, "%Y-%m-%d-%H-%M")
        end = datetime.strptime(self.end_date, "%Y-%m-%d-%H-%M")
        # request_volume = abs((end - start).total_seconds()) / self.granularity
        # time.sleep(10)


        start_unix_timestamp = int(start.timestamp())
        end_unix_timestamp = int(end.timestamp())
        request_volume = abs((end_unix_timestamp - start_unix_timestamp)) / self.granularity

        if request_volume <= 300:
            response = requests.get(
                "https://api.exchange.coinbase.com/products/{0}/candles?start={1}&end={2}&granularity={3}".format(
                    self.ticker,
                start_unix_timestamp,
                end_unix_timestamp,
                    self.granularity))
            if response.status_code in [200, 201, 202, 203, 204]:
                if self.verbose:
                    print('Retrieved Data from Coinbase Pro API.')
                data = pd.DataFrame(json.loads(response.text))
                data.columns = ["time", "low", "high", "open", "close", "volume"]
                data["time"] = pd.to_datetime(data["time"])
                data['time'] = data['time'].apply(lambda x: int(x.timestamp()))
                # data['unix_time'] = data['time'].astype(int) // 10**9  # Convert nanoseconds to seconds

                data = data[data['time'].between(start_unix_timestamp, end_unix_timestamp)]
                data.set_index("time", drop=False, inplace=True)
                data.sort_index(ascending=True, inplace=True)
                data.drop_duplicates(subset=None, keep='first', inplace=True)


                if self.verbose:
                    print('Returning data.')
                return data
            elif response.status_code in [400, 401, 404]:
                if self.verbose:
                    print("Status Code: {}, malformed request to the CoinBase Pro API.".format(response.status_code))
                sys.exit()
            elif response.status_code in [403, 500, 501]:
                if self.verbose:
                    print("Status Code: {}, could not connect to the CoinBase Pro API.".format(response.status_code))
                sys.exit()
            else:
                if self.verbose:
                    print("Status Code: {}, error in connecting to the CoinBase Pro API.".format(response.status_code))
                sys.exit()
        else:
            # The api limit:
            max_per_mssg = 300
            data = pd.DataFrame()
            for i in range(int(request_volume / max_per_mssg) + 1):
                provisional_start = start + timedelta(0, i * (self.granularity * max_per_mssg))
                provisional_start = self._date_cleaner(provisional_start)
                provisional_end = start + timedelta(0, (i + 1) * (self.granularity * max_per_mssg))
                provisional_end = self._date_cleaner(provisional_end)

                print("Provisional Start: {}".format(provisional_start))
                print("Provisional End: {}".format(provisional_end))
                response = requests.get(
                    "https://api.exchange.coinbase.com/products/{0}/candles?start={1}&end={2}&granularity={3}".format(
                        self.ticker,
                        provisional_start,
                        provisional_end,
                        self.granularity))

                if response.status_code in [200, 201, 202, 203, 204]:
                    if self.verbose:
                        print('Data for chunk {} of {} extracted'.format(i+1,
                                                                         (int(request_volume / max_per_mssg) + 1)))
                    dataset = pd.DataFrame(json.loads(response.text))
                    # print(dataset)
                    if not dataset.empty:
                        data = pd.concat([data, dataset], ignore_index=True)
                        time.sleep(randint(0, 2))
                    else:
                        print("""CoinBase Pro API did not have available data for '{}' beginning at {}.  
                        Trying a later date:'{}'""".format(self.ticker,
                                                           self.start_date,
                                                           provisional_start))
                        time.sleep(randint(0, 2))
                elif response.status_code in [400, 401, 404]:
                    if self.verbose:
                        print(
                            "Status Code: {}, malformed request to the CoinBase Pro API.".format(response.status_code))
                    sys.exit()
                elif response.status_code in [403, 500, 501]:
                    if self.verbose:
                        print(
                            "Status Code: {}, could not connect to the CoinBase Pro API.".format(response.status_code))
                    sys.exit()
                else:
                    if self.verbose:
                        print("Status Code: {}, error in connecting to the CoinBase Pro API.".format(
                            response.status_code))
                    sys.exit()
            data.columns = ["time", "low", "high", "open", "close", "volume"]
            data["time"] = pd.to_datetime(data["time"], unit='s')
            data = data[data['time'].between(start, end)]
            data.set_index("time", drop=False, inplace=True)
            data.sort_index(ascending=True, inplace=True)
            data.drop_duplicates(subset=None, keep='first', inplace=True)
            return data

# os.makedirs('data/1m/',exist_ok=True)
# time.sleep(5)
pair = input('Enter Primary Token Pair: EX: BTC-USD: ')
# pair2 = input('Enter Secondary Token Pair: EX: DESO-USD: ')
data = HistoricalData(pair, 60, '2025-01-26-00-00').retrieve_data()
print(data.columns)
data['time'] = data['time'].apply(lambda x: int(x.timestamp()))

data.set_index("time", drop=False, inplace=True)
data.sort_index(ascending=True, inplace=True)
a = data.drop_duplicates(subset=None, keep='first', inplace=True)

data.to_csv('data/1m/{}_currency_high.csv'.format(pair), index=False)

a = pd.read_csv('data/1m/{}_currency_high.csv'.format(pair))
# print(data.columns)






client = Client(api_key, api_secret)


import requests
import json
import base64
import hashlib
import hmac
import time

import os
import pandas as pd 
import librosa
def strip(x, frame_length, hop_length):
    # Compute RMSE.
    rmse = librosa.feature.rms(x, frame_length=frame_length, hop_length=hop_length, center=True)
    # Identify the first frame index where RMSE exceeds a threshold.
    thresh = 0.01
    frame_index = 0
    while rmse[0][frame_index] < thresh:
        frame_index += 1
        
    # Convert units of frames to samples.
    start_sample_index = librosa.frames_to_samples(frame_index, hop_length=hop_length)
    
    # Return the trimmed signal.
    return x[start_sample_index:]

# def latestBlock():
#     web3.eth.getBlock('latest')
#     web3.eth.getBlock('latest')
#     web3.eth.getBlock('latest')
#     web3.eth.getBlock('latest')
#     web3.eth.getBlock('latest')
#     web3.eth.getBlock('latest')
#     web3.eth.getBlock('latest')
#     web3.eth.getBlock('latest')
#     web3.eth.getBlock('latest')
#     web3.eth.getBlock('latest')
#     web3.eth.getBlock('latest')
#     a = web3.eth.getBlock('latest')
#     import time 
#     time.sleep(3)
#     return a
# current_balance = float(client.get_buy_price(currency_pair='ETH-USD')['amount'])* float(auth_client_currency)
'''Buy Me A Coffee :)'''


def profit_target(token,current_holdings,target_percentage): 
    token = token
    print('\n\n {} target'.format(token))
    current_holdings = current_holdings
    target_percentage = current_holdings * float(target_percentage)
    total_target = current_holdings+target_percentage
    print('{} profit target {}, == {}'.format(token,target_percentage,total_target))
    return target_percentage 
def loss(token,current_holdings,loss):
    token = token
    print('\n\n {} loss'.format(token))
    current_holdings = current_holdings
    target_percentage = current_holdings * float(loss)
    total_loss = current_holdings-target_percentage
    print('{} stop loss {}, == {}'.format(token,target_percentage,total_loss)) 
    return target_percentage 

currency = pair


current_price = client.get_buy_price(currency_pair=pair)
print('amount to trade is betweet 0.03<=>0.1 ETH')
auth_client_currency = input('Enter Amount of Crypto to trade from {}'.format(pair))
print('available {} for trading: {}\n\n'.format(currency,auth_client_currency))

current_balance = float(client.get_buy_price(currency_pair=pair)['amount'])* float(auth_client_currency)




print('-->PROFIT TARGETS:')
tar = profit_target(currency,current_balance, .1) 
tar2 = profit_target(currency,current_balance, .01) 
tar3 = profit_target(currency,current_balance, .03) 
print('\n\n -->MAX LOSS:')
loss = loss(currency,current_balance, .1)

print('current balances: {}\n\n'.format(current_balance))

### CBPRO




current_price = client.get_buy_price(currency_pair=pair)
current_price = current_price['amount']
current_price = float(current_price)
current_sell = client.get_sell_price(currency_pair=pair)
current_sell = current_sell['amount']
current_sell = float(current_sell)

# price_data = {'current_sell':current_sell,'current_buy':current_price}
# pricing = Fclient.query(q.create(
#     q.collection('prices'),
#     {'data':price_data}
# ))
# print('buy price', current_price)
# print('sell price', current_sell)


'''INITIALIZE DEEP LEARNING WITH LIVE DATA'''
def deeper():
    import pandas as pd 
    import matplotlib.pyplot as plt
    # import IPython.display as ipd
    import pandas as pd
    import librosa
    # import keras
    import librosa.display
    import time
    import glob

    # import plotly.express as px
    from sklearn.decomposition import PCA, FastICA
    warnings.filterwarnings('ignore')

    if not os.path.exists("images"):
        os.mkdir("images")
        
    '''call chosen currency via coinbase API'''
    iteration=1
    import time
    time.sleep(1) 

    current_price = client.get_buy_price(currency_pair=pair)
    current_price = float(current_price['amount'])
    current_sell = client.get_sell_price(currency_pair=pair)
    current_sell = current_sell['amount']
    current_sell = float(current_sell)

    # price_data = {'current_sell':current_sell,'current_buy':current_price}
    # pricing = Fclient.query(q.create(
    #     q.collection('prices'),
    #     {'data':price_data}
    # ))
    # pricing
    # print('buy price', current_price)
    # print('sell price', current_sell)
    amount = auth_client_currency
    currency = pair
    
    current_balance = float(client.get_buy_price(currency_pair='{}'.format(pair))['amount'])* float(auth_client_currency)
    print('current balances: {}\n\n'.format(current_balance))
    '''Call Chosen Currency History'''
#     data = pd.read_csv('1m/ETH-USD.csv')  
#     data = currency
    data = HistoricalData(pair, 60, '2025-01-26-00-00').retrieve_data()
    print(data.columns)
    data['time'] = data['time'].apply(lambda x: int(x.timestamp()))

    data.set_index("time", drop=False, inplace=True)
    data.sort_index(ascending=True, inplace=True)
    data.drop_duplicates(subset=None, keep='first', inplace=True)
    period = 14 
    data['RSI (NATIVE)'] = RSIIndicator(data['open'],window=period).rsi()
    data['RSI (NATIVE)'] = data['RSI (NATIVE)'].fillna(0.0)
    data.to_csv('data/1m/{}_currency_high.csv'.format(pair),index=False)
    a0 = pd.read_csv('data/1m/{}_currency_high.csv'.format(pair))


    '''Isolate Features From Respective Currency Data'''
    # a0 = a0.drop(['Unnamed: 0'], axis=0 )
    b0 = a0['open']
    c0 = a0['high']
    d0 = a0['low']
    e0 = a0['close']
    f0 = a0['volume']
    i0 = a0['time']

    # price_data = {'current_sell':current_sell,'current_buy':current_price,'solTradePrice (Open)':float(b0[-1:])}
    # pricing = Fclient.query(q.create(
    #     q.collection('prices'),
    #     {'data':price_data}
    # ))
    # pricing
    print('buy price', current_price)
    print('sell price', current_sell)

#     '''Visualize Interactive Data Via plotly'''
#     fig = go.Figure(data=[go.Candlestick(x=a0, 
#                          open=b0, 
#                          high=c0, 
#                          low=d0, 
#                          close=e0)])
#     print('Use The Slider to Adjust and Zoom')
#     fig.show()
    
    # order_book = client.get_orders
    # print('Order Book \n\n',order_book)
    
    '''Averaging Isolated Price Data'''
    avg=np.average(b0)   
    avg1=np.average(c0) 
    avg2=np.average(d0)
    avg3=np.average(e0)
    print('avg OPEN : {}, avg High : {}, avg LOW : {}, avg CLOSE : {}\n\n'.format(avg,avg1,avg2,avg3)) 

    '''Display x=Volume , y = Open Signals '''
#     data = dict(
#         number=[b0,c0,d0,e0],
#         stage=[ "Open", "High", "Low", "Close"])
#     fig = px.funnel(a0, x=f0, y=b0)
#     fig.show()

    '''Display x=Time , y = Open Signals '''
#     data = dict(
#         number=[b0,c0,d0,e0],
#         stage=[ "Open", "High", "Low", "Close"])
#     fig = px.funnel(a0, x=i0, y=b0)
#     fig.show()
    
    '''currency volume'''
    background = f0
    '''Time'''
    x = i0
    '''Open'''
    y = b0
    '''Creating isolated datasets'''
    x_df = pd.DataFrame(x)  
    y_df = pd.DataFrame(y) 
    background_df = pd.DataFrame(background) 
    x = x_df 
    y = y_df 

    '''Extract and rejoin volume,time,open data'''
    background = background_df
    extract = x.join(background) 
    extract = extract.join(y)
    extract 
       
    data = extract.to_csv('data/1m/{}_extraction_data.csv'.format(pair)) 
    data = pd.read_csv('data/1m/{}_extraction_data.csv'.format(pair))
    data = data.drop(['Unnamed: 0'],axis=1) 
    data 

    X= i0 #time
    y = data['open'] 
    background = data['volume']

#     plt.plot(y) 
#     # plt.plot(x)
#     plt.hist2d(i0,data['open']) 
#     plt.hist2d(i0,data['volume']) 

    '''Restructure data so algorithim can read data and udate sample rates for linear regression '''
    data = np.squeeze(np.asarray(np.matrix(data)[:,1])) 
    sam_rate = np.squeeze(np.asarray(np.matrix(data)[:,0])) 
    D = np.abs(librosa.stft(data))**2
    # S = librosa.feature.melspectrogram(data,sr=sam_rate,S=D,n_mels=512)
    # log_S1 = librosa.power_to_db(S,ref=np.max)

# #     plt.figure(figsize=(12,4))
# #     librosa.display.specshow(log_S1,sr=sam_rate,x_axis='time',y_axis='mel')
# #     plt.title('MEL POWER SPECTOGRAM')

# #     plt.colorbar(format='%+02.0f dB')

# #     plt.tight_layout()
# #     plt.show()
    # librosa.get_duration(data, sam_rate)
    # h_l = 500
    # f_l = 0
    h_l = 256 
    f_l = 512


    rsi = a0['RSI (NATIVE)'][-1:] 
    rsi = float(rsi)
    if current_sell >= current_price or rsi>=58.00: # or current_price >= current_sell or 
        import time
        print('SELLING')

    #     # Define hyperparameters
        batch_size = 32
        time_steps = 60  # Number of previous time steps to consider for prediction
        epochs = 3
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data.reshape(-1, 1))

        # Split the data into training and testing sets
        training_data_len = math.ceil(len(data) * 0.8)
        train_data, test_data = scaled_data[0:training_data_len, :], scaled_data[training_data_len - time_steps:, :]


        xl_train, yl_train = [], []
        # print(yh[-1:])
        Hdata = a0['high'].values
        xh_train, yh_train = [], []
        for i in range(time_steps, len(train_data)):
            xh_train.append(train_data[i - time_steps:i, 0])
            yh_train.append(train_data[i, 0])

        xh_train, yh_train = np.array(xh_train), np.array(yh_train)
        xh_train = np.reshape(xh_train, (xh_train.shape[0], xh_train.shape[1], 1))

        #  LSTM model
        # model = Sequential()
        # model.add(LSTM(5, return_sequences=True, input_shape=(xh_train.shape[1], 1)))
        # model.add(LSTM(1, return_sequences=False))
        
        # model.add(Dense(1))
        ######################
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, input_shape=(xh_train.shape[1], 1)))
        model.add(Dropout(.01))
        model.add(LSTM(32, return_sequences=False))
        # model.add(Dropout(0.2))
        model.add(Dense(1))
 
        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error',learning_rate=0.62)

        # Train the model
        model.fit(xh_train, yh_train, batch_size=batch_size, nb_epoch=3)

        # testing dataset
        xh_test, yh_test = [], []
        for i in range(time_steps, len(test_data)):
            xh_test.append(test_data[i - time_steps:i, 0])
            yh_test.append(test_data[i, 0])

        xh_test, yh_test = np.array(xh_test), np.array(yh_test)
        xh_test = np.reshape(xh_test, (xh_test.shape[0], xh_test.shape[1], 1))

        # Make predictions
        H_predictions = model.predict(xh_test)
        print(H_predictions[-1:])
        yh = scaler.inverse_transform(H_predictions)
        print(yh[-1:])

        xh_FUTURE =  25
        H_predictions = np.array([])
        H_Last = xh_test[-1]
        for i in range(xh_FUTURE):
            curr_prediction_L = model.predict(np.array([H_Last]))
            print(curr_prediction_L)
            H_Last = np.concatenate([H_Last[1:], curr_prediction_L])
            H_predictions = np.concatenate([H_predictions, curr_prediction_L[0]])
            
        minimum = np.min(Hdata[-75:])    
        minimum

        maximum = np.max(Hdata[-75:])    
        maximum

        print(maximum)

        H_predictions = H_predictions * minimum + (maximum - minimum)
        print(H_predictions)
    
        timestamp = str(int(time.time()))
        conn = http.client.HTTPSConnection("api.coinbase.com")
        method = "POST"
        path = "/api/v3/brokerage/orders"
        int_order_id = np.random.randint(4**12)
        #Predicted high
        last_high = str(yh[-1:])
        last_low = yh[-1:]
        print('last high', last_high)
        print('last low', last_low)
        # last_low = str(round(float(last_low), 2))
        #Actual Previous High 
        #yh_test[-1:]
        # last_high = str(round(float(last_high), 3))


        risk_tolerance = 0.02  # Adjust this value to your desired risk tolerance
        stop_loss_price = last_low * risk_tolerance
        # time.sleep(2)

        payload = "{\"client_order_id\":" + "\"" + str(int_order_id) + "\"" + ",\"product_id\":"+"\""+pair+ "\",\"side\":\"SELL\",\"order_configuration\": {\"limit_limit_gtc\": {\"base_size\":\""+str(auth_client_currency)+"\",\"limit_price\":\"" + str(current_sell) + "\"}}}"  #TODO Update base_size to amount to buy#{\"market_market_ioc\": {\"base_size\":\"1250\"}}

        message = timestamp + method + path.split('?')[0] + str(payload)
        signature = hmac.new(api_secret.encode('utf-8'), message.encode('utf-8'), digestmod=hashlib.sha256).hexdigest()

        headers={
            'CB-ACCESS-KEY': api_key,
            'CB-ACCESS-TIMESTAMP': timestamp,
            'CB-ACCESS-SIGN': signature,
            'accept':'application/json'
            }

        print(payload)

        conn.request(method, path, payload, headers)
        res = conn.getresponse()
        data = res.read()
        print(data.decode("utf-8"))


        if current_price <= stop_loss_price:
            print("Market conditions triggered a stop-loss. Exiting the trade.")
            # print('current price higher than previous selling at predicted price {}'.format(yh[-1:]))
            print('current price higher than previous/ selling at predicted price {}'.format(yh[-1:]))
            print('current trading balance ',current_balance)
            limit_price = yh[-1:]
            limit_price = float(limit_price[0])  # Convert to float
            limit_price = str('{:.2f}'.format(limit_price))

            payload = "{\"client_order_id\":" + "\"" + str(int_order_id) + "\"" + ",\"product_id\":"+"\""+pair+ "\",\"side\":\"SELL\",\"order_configuration\": {\"limit_limit_gtc\": {\"base_size\":\""+str(auth_client_currency)+"\",\"limit_price\":\"" + str(limit_price) + "\"}}}"  #TODO Update base_size to amount to buy#{\"market_market_ioc\": {\"base_size\":\"1250\"}}

            message = timestamp + method + path.split('?')[0] + str(payload)
            signature = hmac.new(api_secret.encode('utf-8'), message.encode('utf-8'), digestmod=hashlib.sha256).hexdigest()

            headers={
            'CB-ACCESS-KEY': api_key,
            'CB-ACCESS-TIMESTAMP': timestamp,
            'CB-ACCESS-SIGN': signature,
            'accept':'application/json'
            }

            print(payload)

            conn.request(method, path, payload, headers)
            res = conn.getresponse()
            data = res.read()
            print(data.decode("utf-8"))

        if current_sell >= current_price or rsi>=58.00 or float(bot_limit[0]) >= current_price*.3:
            print('current price higher than previous/ selling at predicted price {}'.format(yh[-1:]))
            print('current trading balance ',current_balance)
            # limit_price = yh[-1:]
            # limit_price = float(limit_price[0])  # Convert to float
            # limit_price = str('{:.2f}'.format(limit_price))


            limit_price =  str('{:.2f}'.format(current_sell))

            bot_limit = yh[-1:]
            bot_limit = float(bot_limit[0])  # Convert to float
            bot_limit = str('{:.2f}'.format(bot_limit))

            payload = "{\"client_order_id\":" + "\"" + str(int_order_id) + "\"" + ",\"product_id\":"+"\""+pair+ "\",\"side\":\"SELL\",\"order_configuration\": {\"limit_limit_gtc\": {\"base_size\":\""+str(auth_client_currency)+"\",\"limit_price\":\"" + str(limit_price) + "\"}}}"  #TODO Update base_size to amount to buy#{\"market_market_ioc\": {\"base_size\":\"1250\"}}

            message = timestamp + method + path.split('?')[0] + str(payload)
            signature = hmac.new(api_secret.encode('utf-8'), message.encode('utf-8'), digestmod=hashlib.sha256).hexdigest()

            headers={
            'CB-ACCESS-KEY': api_key,
            'CB-ACCESS-TIMESTAMP': timestamp,
            'CB-ACCESS-SIGN': signature,
            'accept':'application/json'
            }

            print(payload)

            conn.request(method, path, payload, headers)
            res = conn.getresponse()
            data = res.read()
            print(data.decode("utf-8"))
            time.sleep(1)
            payload = "{\"client_order_id\":" + "\"" + str(int_order_id) + "\"" + ",\"product_id\":"+"\""+pair+ "\",\"side\":\"SELL\",\"order_configuration\": {\"limit_limit_gtc\": {\"base_size\":\""+str(auth_client_currency)+"\",\"limit_price\":\"" + str(bot_limit) + "\"}}}"  #TODO Update base_size to amount to buy#{\"market_market_ioc\": {\"base_size\":\"1250\"}}

            message = timestamp + method + path.split('?')[0] + str(payload)
            signature = hmac.new(api_secret.encode('utf-8'), message.encode('utf-8'), digestmod=hashlib.sha256).hexdigest()

            headers={
            'CB-ACCESS-KEY': api_key,
            'CB-ACCESS-TIMESTAMP': timestamp,
            'CB-ACCESS-SIGN': signature,
            'accept':'application/json'
            }

            print(payload)

            conn.request(method, path, payload, headers)
            res = conn.getresponse()
            data = res.read()
            print(data.decode("utf-8"))


          
    
    if current_price <= current_sell or rsi>=65.00: #current_price <= current_sell or 
        print('BUYING')
        data = a0['low'].values

        # Define hyperparameters
        batch_size = 12
        time_steps = 60  # Number of previous time steps to consider for prediction
        epochs = 3

        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data.reshape(-1, 1))

        # Split the data into training and testing sets
        training_data_len = math.ceil(len(data) * 0.8)
        train_data, test_data = scaled_data[0:training_data_len, :], scaled_data[training_data_len - time_steps:, :]

        xl_train, yl_train = [], []
        for i in range(time_steps, len(train_data)):
            xl_train.append(train_data[i - time_steps:i, 0])
            yl_train.append(train_data[i, 0])

        xl_train, yl_train = np.array(xl_train), np.array(yl_train)
        xl_train = np.reshape(xl_train, (xl_train.shape[0], xl_train.shape[1], 1))

        # LSTM model
        model = Sequential()
        model.add(LSTM(5, return_sequences=True, input_shape=(xl_train.shape[1], 1)))
        model.add(LSTM(1, return_sequences=False))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error',learning_rate=0.78)

        # model = Sequential()
        # model.add(LSTM(64, return_sequences=True, input_shape=(xl_train.shape[1], 1)))
        # model.add(Dropout(0.2))
        # model.add(LSTM(64, return_sequences=False))
        # model.add(Dropout(0.2))
        # model.add(Dense(1))
 
        model.compile(optimizer='adam', loss='mean_squared_error',learning_rate=0.018)

        # Train model
        model.fit(xl_train, yl_train, batch_size=batch_size, nb_epoch=3)


        xl_test, yl_test = [], []
        for i in range(time_steps, len(test_data)):
            xl_test.append(test_data[i - time_steps:i, 0])
            yl_test.append(test_data[i, 0])

        xl_test, yl_test = np.array(xl_test), np.array(yl_test)
        xl_test = np.reshape(xl_test, (xl_test.shape[0], xl_test.shape[1], 1))

        # Make predictions
        L_predictions = model.predict(xl_test)
        print(L_predictions[-1:])
        yl = scaler.inverse_transform(L_predictions)
        print(yl[-1:])

        Xl_FUTURE =  25
        L_predictions = np.array([])
        L_last = xl_test[-1]
        for i in range(Xl_FUTURE):
            curr_prediction_L = model.predict(np.array([L_last]))
            print(curr_prediction_L)
            L_last = np.concatenate([L_last[1:], curr_prediction_L])
            L_predictions = np.concatenate([L_predictions, curr_prediction_L[0]])
            
        minimum = np.min(data[-75:])    
        minimum

        maximum = np.max(data[-75:])    
        maximum

        print(maximum)

        L_predictions = L_predictions * minimum + (maximum - minimum)
        print(L_predictions)
        

        timestamp = str(int(time.time()))
        conn = http.client.HTTPSConnection("api.coinbase.com")
        method = "POST"
        path = "/api/v3/brokerage/orders"
        int_order_id = np.random.randint(4**6)
        last_low = yl[-1:]
        last_low = float(last_low[0])  # Convert to float
        last_low = str('{:.2f}'.format(last_low))
        payload = "{\"client_order_id\":" + "\"" + str(int_order_id) + "\"" + ",\"product_id\":"+"\""+pair+ "\",\"side\":\"BUY\",\"order_configuration\": {\"limit_limit_gtc\": {\"base_size\":\""+str(auth_client_currency)+"\",\"limit_price\":\"" + last_low + "\"}}}"


        message = timestamp + method + path.split('?')[0] + str(payload)
        signature = hmac.new(api_secret.encode('utf-8'), message.encode('utf-8'), digestmod=hashlib.sha256).hexdigest()

        
        headers={
        'CB-ACCESS-KEY': api_key,
        'CB-ACCESS-TIMESTAMP': timestamp,
        'CB-ACCESS-SIGN': signature,
        'accept':'application/json'
        }

        print(payload)


        conn.request(method, path, payload, headers)
        res = conn.getresponse()
        data = res.read()
        print(data.decode("utf-8"))
        time.sleep(1)
        payload = "{\"client_order_id\":" + "\"" + str(int_order_id) + "\"" + ",\"product_id\":"+"\""+pair+ "\",\"side\":\"BUY\",\"order_configuration\": {\"limit_limit_gtc\": {\"base_size\":\""+str(auth_client_currency)+"\",\"limit_price\":\"" + str(current_price) + "\"}}}"


        message = timestamp + method + path.split('?')[0] + str(payload)
        signature = hmac.new(api_secret.encode('utf-8'), message.encode('utf-8'), digestmod=hashlib.sha256).hexdigest()

        
        headers={
        'CB-ACCESS-KEY': api_key,
        'CB-ACCESS-TIMESTAMP': timestamp,
        'CB-ACCESS-SIGN': signature,
        'accept':'application/json'
        }

        print(payload)


        conn.request(method, path, payload, headers)
        res = conn.getresponse()
        data = res.read()
        print(data.decode("utf-8"))




iteration = 0
while iteration < 100:  # Add a maximum number of iterations as a condition
    deeper()
    iteration += 1
    time.sleep(1)  # Add a sleep interval to avoid constant API requests
