import alpaca_trade_api as tradeapi
import requests
import time
import json
import ta.volatility as ta
from ta import momentum
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression as lr
from datetime import datetime, timedelta
from pytz import timezone
from threading import Thread
import time
import copy
import pytz
from yahoo_fin import stock_info as si

"""
An attempt to make a TTM Squeeze Indicator
"""


#read credentials in from json file

cred = json.load(open("credentials.json"))
base_url = cred['endpoint']
api_key_id = cred['key']
api_secret = cred['secret']
mailgun = cred['mailgun']
mailgunURL = cred['URL']
email = cred['email']




api = tradeapi.REST(
    base_url=base_url,
    key_id=api_key_id,
    secret_key=api_secret
)

session = requests.session()

#sqzonl = False
#sqzoffl = False
#nosqzl = True
sqzl = {}
sqzhl = {}

#times to aggregate to (have to manually do 5 minute increments because polygon doesnt allow 5 min for some reason)
ftimes = "08:30,08:35,08:40,08:45,08:50,08:55,09:00,09:05,09:10,09:15,09:20,09:25,09:30,09:35,09:40,09:45,09:50,09:55,10:00,10:05,10:10,10:15,10:20,10:25,10:30,10:35,10:40,10:45,10:50,10:55,11:00,11:05,11:10,11:15,11:20,11:25,11:30,11:35,11:40,11:45,11:50,11:55,12:00,12:05,12:10,12:15,12:20,12:25,12:30,12:35,12:40,12:45,12:50,12:55,13:00,13:05,13:10,13:15,13:20,13:25,13:30,13:35,13:40,13:45,13:50,13:55,14:00,14:05,14:10,14:15,14:20,14:25,14:30,14:35,14:40,14:45,14:50,14:55,15:00".split(",")


min_share_price = 5.0
max_share_price = 50.0
# Minimum previous-day dollar volume for a stock we might consider
min_last_dv = 2500000
# Stop limit to default to
#default_stop = .95



def get_hour_historical(symbols_h):
    
    mins = {}
    hour = {}

    count = 0
    print("min5_history")
    for s in symbols_h:
        s = s.ticker
        truemins = api.polygon.historic_agg(size="minute", symbol=s, limit=1260).df
        truehourh = copy.deepcopy(truemins)
        count+=1

        mins[s] = copy.deepcopy(truemins)
        truemins['volume'] = mins[s]['volume'].dropna().resample('5min', loffset='5min').sum()
        truemins['open'] = mins[s]['open'].dropna().resample('5min', loffset='5min').first()
        truemins['close']= mins[s]['close'].dropna().resample('5min', loffset='5min').last()
        truemins['high'] = mins[s]['high'].dropna().resample('5min', loffset='5min').max()
        truemins['low'] = mins[s]['low'].dropna().resample('5min', loffset='5min').min()
        #print(truemins)
        mins[s] = truemins.dropna().resample('5min', loffset='5min').sum()
        

        hour[s] = copy.deepcopy(truemins)
        truehourh['volume'] = hour[s]['volume'].dropna().resample('60min', loffset='30min').sum()
        truehourh['open'] = hour[s]['open'].dropna().resample('60min', loffset='30min').first()
        truehourh['close']= hour[s]['close'].dropna().resample('60min', loffset='30min').last()
        truehourh['high'] = hour[s]['high'].dropna().resample('60min', loffset='30min').max()
        truehourh['low'] = hour[s]['low'].dropna().resample('60min', loffset='30min').min()
        hour[s] = truehourh.dropna().resample('60min', loffset='30min').sum()

        #index = test[test[]]
        
        td = []
        for i in hour[s].index:
            for g in ftimes:
                if str(i)[11:16] not in ftimes:
                    td.append(i)
        hour[s] = hour[s].drop(td)
        hour[s] = hour[s].drop(hour[s][hour[s]['volume'] == 0].index)
        td = []
        for i in mins[s].index:
            for g in ftimes:
                if str(i)[11:16] not in ftimes:
                    td.append(i)
        mins[s] = mins[s].drop(td)
        mins[s] = mins[s].drop(mins[s][mins[s]['volume'] == 0].index)
    

    return mins, hour

"""
def find_stop(current_value, min5_history, now):
    series = min5_history['low'][-100:] \
                .dropna().resample('5min').min()
    series = series[now.floor('1D'):]
    diff = np.diff(series.values)
    low_index = np.where((diff[:-1] <= 0) & (diff[1:] > 0))[0] + 1
    if len(low_index) > 0:
        return series[low_index[-1]] - 0.01
    return current_value * default_stop

target_prices[symbol] = data.close + (
                (data.close - stop_price) * 3
            )
"""

def run(tickers):
    # steam connection
    
    conn = tradeapi.StreamConn(base_url=base_url, key_id=api_key_id, secret_key=api_secret)

    # Update initial state with information from tickers
    
    symbols = tickers

    #initialize sqz states

    for s in tickers:
        sqzl[s.ticker] = [False, False, True]
        sqzhl[s.ticker] = [False, False, True]

    #minute_history = get_historical([symbol])
    min5_history, hour_history = get_hour_historical(symbols)

    print('running')
    #print (hour_history)

    # Use trade updates to keep track of our portfolio
    
    @conn.on('trade_update')
    async def handle_trade_update(conn, channel, data):
        print("hi")

    @conn.on('A.*')
    async def handle_second_bar(conn, channel, data):
        symbol = data.symbol
        
        #global sqzonl
        #global sqzoffl
        #global nosqzl

        global sqzl
        global sqzhl

        #print(data.symbol)

        #add latest second data to 5min dataframe  
        ts = data.start
        tm = ts - timedelta(hours = 1, minutes= ts.minute%5, seconds=ts.second, microseconds=ts.microsecond)
        try:
            current = min5_history[data.symbol].loc[ts]
        except KeyError:
            current = None
        new_data = []
        if current is None:
            new_data = [
                data.open,
                data.high,
                data.low,
                data.close,
                data.volume
            ]
        else:
            new_data = [
                current.open,
                data.high if data.high > current.high else current.high,
                data.low if data.low < current.low else current.low,
                data.close,
                current.volume
            ]
    
        min5_history[symbol].loc[tm] = new_data


        #add latest second data to hour dataframe
        th = ts - timedelta(hours = 1 - ts.minute//30,minutes= ts.minute - 30, seconds=ts.second, microseconds=ts.microsecond)

        try:
            current = hour_history[data.symbol].loc[ts]
        except KeyError:
            current = None
        new_data = []
        if current is None:
            new_data = [
                data.open,
                data.high,
                data.low,
                data.close,
                data.volume
            ]
        else:
            new_data = [
                current.open,
                data.high if data.high > current.high else current.high,
                data.low if data.low < current.low else current.low,
                data.close,
                current.volume
            ]
        hour_history[symbol].loc[th] = new_data

        #print(min5_history[symbol].loc[ts])
        
        thread = Thread(target=squeeze, args=(min5_history, sqzl, symbol, "5min"))
        squeeze(min5_history, sqzl, symbol, "5min")
        squeeze(hour_history, sqzhl, symbol, "hour")

        if(not api.get_clock().is_open):
            symbols.remove(symbol)
            if len(symbols) <= 0:
                conn.close()
                start()
            conn.deregister([
            'AM.{}'.format(symbol),
            'A.{}'.format(symbol)
    ])
        

    # Replace aggregated 1s bars with incoming 1m bars

    #IDEA: RUN A MINUTE BASED TTM SQUEEZE ALGORITHM AND AN HOUR BASED ALGORITHM.
    #CURRENT: Aggregates new minute bar data to help resolve any potential data loss from the second based data.
    @conn.on('AM.*')
    async def handle_minute_bar(conn, channel, data):
        ts = data.start

        #print(min5_history)
        #print(hour_history)
        
        #5 Minute Data
        tm = ts - timedelta(hours = 1, minutes= ts.minute%5, seconds=ts.second, microseconds=ts.microsecond)
        print(tm)
        current = min5_history[data.symbol].loc[tm]
        min5_history[data.symbol].loc[tm] = [
            data.open,
            data.high if data.high > current.high else current.high,
            data.low if data.low < current.low else current.low,
            data.close,
            data.volume+current.volume
        ]


        #Hour Data
        th = ts - timedelta(hours = 1 - ts.minute//30,minutes= ts.minute - 30, seconds=ts.second, microseconds=ts.microsecond)
        print(th)
        current = hour_history[data.symbol].loc[th]
        hour_history[data.symbol].loc[th] = [
            data.open,
            data.high if data.high > current.high else current.high,
            data.low if data.low < current.low else current.low,
            data.close,
            data.volume+current.volume
        ]
        print('------------------------------------hour------------------------------------')
        print(hour_history[data.symbol].loc[th])
        print('------------------------------------mins------------------------------------')
        print(min5_history[data.symbol].loc[tm])
        print("----------------------------------------------------------------------------")
        print(min5_history)
        print(ts, tm, th)

        #print(min5_history)
        #print(hour_history)
    
    #have to aggregate minute data into hour based data.
    #

    channels = ['trade_updates']
    for s in symbols:
        print(s.ticker)
        symbol_channels = ['A.{}'.format(s.ticker), 'AM.{}'.format(s.ticker)]
        channels += symbol_channels
    print("watching {} symbols".format(len(symbols)))
    print(channels)
    run_ws(conn, channels)


#Handle failed websocket connections by reconnecting 
def run_ws(conn, channels):
    try:
        conn.run(channels)
    except Exception as e:
        print(e)
        conn.close
        run_ws(conn, channels)

def send_message(symbol, action):
    n = datetime.now()
    n = n - timedelta(microseconds=n.microsecond)
    n = n.time()
    return requests.post(
	"https://api.mailgun.net/v3/"+ mailgunURL +"/messages",
	auth=("api", mailgun),
	data={"from": "TRADE NOTIFICATION<postmaster@"+ mailgunURL +">",
			"to": email,
			"subject": "{} - {}".format(symbol, action),
			"text": "Time: {}".format(n)})

def squeeze(history, sqz, symbol, t):
    bbh = ta.bollinger_hband(history[symbol]['close'].dropna(), n = 20, ndev=2)
    bbl = ta.bollinger_lband(history[symbol]['close'].dropna(), n = 20, ndev=2)
    bba = ta.bollinger_mavg(history[symbol]['close'].dropna(), n=20)

    kca = ta.keltner_channel_central(history[symbol]['high'],history[symbol]['low'], history[symbol]['close'], n=20)
    kcl = ta.keltner_channel_lband(history[symbol]['high'],history[symbol]['low'], history[symbol]['close'], n=20)
    kch = ta.keltner_channel_hband(history[symbol]['high'],history[symbol]['low'], history[symbol]['close'], n=20)
    
    mom = momentum.ao(history[symbol]['high'], history[symbol]['low'])
    
    #print(bbl[-1], kcl[-1], bbh[-1], kch[-1])
    
    sqzon = (bbl[-1]>kcl[-1]) and (bbh[-1]<kch[-1])
    sqzoff = bbl[-1]<kcl[-1] and bbh[-1]>kch[-1]
    nosqz = sqzon==False and sqzoff==False
    
    """
    value = (Highest[lengthKC](high)+Lowest[lengthKC](low)+average[lengthKC](close))/3
    val = linearregression[lengthKC](close-value)

    val = (max(history[symbol]['high'][-20:-1]) + min(history[symbol]['low'][-20:-1]) + history[symbol]['close'].mean())/3.0
    v = lr(history[symbol]['close'] - val)
    """

    flag = -1
    #print("hi")
    if sqz[symbol][0] and sqzon:
        pass

    if sqz[symbol][0] and sqzoff:
        sqz[symbol][0] = False
        sqz[symbol][1] = True
        flag = 0

    if sqz[symbol][1] and sqzoff:
        pass

    if sqz[symbol][1] and sqzon:
        sqz[symbol][1] = False
        sqz[symbol][0] = True
        flag = 1

    if sqz[symbol][2] and sqzon:
        sqz[symbol][2] = False
        sqz[symbol][0] = True
        flag = 1

    if sqz[symbol][2] and sqzoff:
        sqz[symbol][2] = False
        sqz[symbol][1] = True
        #print("sqeeze is OFF")
        #print(time.time())
        flag = -1


    if flag == -1:
        #print('No Change')
        pass
    if flag == 0:
        send_message(symbol, t + " pop : " + str(mom[-1]))

    if flag == 1:
        send_message(symbol, "Squeeze On "+ t)

def get_tickers():
    print('Getting current ticker data...')
    tickers = api.polygon.all_tickers()

    print('Success.')
    assets = api.list_assets()
    symbols = [asset.symbol for asset in assets if asset.tradable]
    
    return [ticker for ticker in tickers if (ticker.ticker in ['AMD', 'PINS', "SQ", 'TSLA', 'LULU', 'WORK', 'KO', 'ATVI', 'SHOP', 'PYPL', 'KO', 'SNAP', 'WORK'])]
    """
    return [ticker for ticker in tickers if (
        ticker.ticker in symbols and
        ticker.lastTrade['p'] >= min_share_price and
        ticker.lastTrade['p'] <= max_share_price and
        ticker.prevDay['v'] * ticker.lastTrade['p'] > min_last_dv #and
        #ticker.todaysChangePerc >= 3.5
    )]
    """

def get_historical(symbols_h):
    print("History")

    minn = {}
    day = {}
    coun = 0
    print(symbols_h)
    for s in symbols_h:
        
        day[s] = api.polygon.historic_agg(size="hour", symbol=s, limit=2160).df
        coun+=1
    print("{}/{}".format(coun, len(symbols_h)))
    print(day[symbols_h[0]])
    return day


def start():
    t = api.get_clock()
    #if t.is_open == False:
    #        tillopen = (t.next_open - t.timestamp).total_seconds()
    #        print("market closed. Sleep for ", int(tillopen)-60, " seconds")
    #        time.sleep(int(tillopen))
        
    #print(sym)
    #print('hi')
    
    amd = si.get_data('AMD', interval='1h')

    #print(get_historical(['AMD', 'SPY', 'GPRO', 'GE', 'WM', 'TWTR', 'SQ', 'KO', 'DIS', 'MSFT', 'SNAP']))

    #run(get_tickers())

if __name__ == "__main__":
    #run(get_tickers())
    start()
    
