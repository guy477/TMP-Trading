import alpaca_trade_api as tradeapi
import requests, time, json
import pandas as pd
import numpy as np
from pytz import timezone
from collections import *
import joblib
from datetime import datetime, timedelta
from pytz import timezone
from sklearn.utils import check_random_state


cred = json.load(open("credentials.json"))
# files = json.load(open("files.json"))


base_url = cred['endpoint']
api_key_id = cred['key']
api_secret = cred['secret']
mailgun = cred['mailgun']
mailgunURL = cred['URL']
email = cred['email']
#Connecte to Alpaca Trade API


api = tradeapi.REST(
    base_url=base_url,
    key_id=api_key_id,
    secret_key=api_secret
)
session = requests.session()


def ema(length, data):
    return data.ewm(span=length, adjust=False, min_periods=length-1).mean()

def getHistorical(period, data):
    hist = []
    conv = []
    o = -1
    vals = [o, -1, float('inf'), -1, -1]
    s = 0
    dupeCount = 0
    for ind, p in data.iterrows():
        # get the open, close, min, and max for each volume period given the minute based data.
        if(vals[0] == -1 or vals[0] == 0):
            vals[0] = p['open']
        s += np.log(p['volume'])
        vals = [vals[0], -1, min(vals[2], p['low']), max(vals[3], p['high']), s]
        if(s > 2*period):
            dupeCount += 1
        if (s >= period):
            
            dif = s - period 
            vals[1] = p['close']
            vals[-1] = period
            hist.append(vals)
            if(dif!=0):
                o = p['close']
                s = period % dif
            else:
                o = -1
                s = 0
            vals = [o, -1, float('inf'), -1, s]
            
            
    # Make sure to catch the last data point, even if it isn't full.
    if((vals[1] == -1)):
        hist.append(vals)

    #print(str(dupeCount) + " condensed pointes for period " + str(period))
    #print(str(len(hist)) + " number of points for period " + str(period))
    hist = pd.DataFrame(hist, columns = ['open', 'close', 'low', 'high', 'volume'])
    return hist

def run(tickers, mult, market_open):
       
    # stream connection
    conn = tradeapi.StreamConn(base_url=base_url, key_id=api_key_id, secret_key=api_secret)
    # Update initial state with information from tickers

    BullHMM = joblib.load('models/-BULL43-6-1.pkl')
    #BearHMM = joblib.load('models/bestBear.pkl')
    scale = joblib.load('scales/-BULL-43.pkl')
    symbols = tickers

    dataPoints = defaultdict(pd.DataFrame)
    pos = {}
    orders = {}
    partials = {}
    roll = defaultdict(pd.DataFrame)
    dif = {}
    pfv = float(api.get_account().portfolio_value)

    print('starting pfv value - '+str(pfv))
    print('running')
    
    
    @conn.on(r'trade_updates')
    async def handle_trade_update(conn, channel, data):
        print('------------------update------------------')
        symbol = data.order['symbol']
        last_order = orders.get(symbol)
        if last_order is not None:
            event = data.event
            if event == 'partial_fill':
                qty = float(data.order['filled_qty'])
                if data.order['side'] == 'sell':
                    qty = qty * -1
                pos[symbol] = (
                    pos.get(symbol, 0) - partials.get(symbol, 0)
                )
                partials[symbol] = qty
                pos[symbol] += qty
                orders[symbol] = data.order
            elif event == 'fill':
                qty = float(data.order['filled_qty'])
                if data.order['side'] == 'sell':
                    qty = qty * -1
                pos[symbol] = (
                    pos.get(symbol, 0) - partials.get(symbol, 0)
                )
                partials[symbol] = 0
                pos[symbol] += qty
                orders[symbol] = None
            elif event == 'canceled' or event == 'rejected':
                partials[symbol] = 0
                orders[symbol] = None
    
    # Second-based data.
    @conn.on('A.*')
    async def handle_second_bar(conn, channel, data):
        symbol = data.symbol
        curMod = None
        # First, aggregate 1s bars for up-to-date MACD calculations
        ts = data.start
        #print(data.symbol)
        #ts -= timedelta(seconds=ts.second, microseconds=ts.microsecond)
        try:
            current = dataPoints[data.symbol].loc[ts]
        except KeyError:
            current = None
        new_data = []
        
        if current is None:
            new_data = [
                data.open,
                data.close,
                data.low,
                data.high,
                np.log(data.volume)
            ]
        
        else:
            new_data = [
                current.open,
                data.close,
                data.low if data.low < current.low else current.low,
                data.high if data.high > current.high else current.high,
                current.volume + np.log(data.volume)
            ]
        
        if(list(dataPoints[symbol].columns) == []):
            dataPoints[symbol] = pd.DataFrame(columns=['open', 'close', 'low', 'high', 'volume'])
        dataPoints[symbol].loc[ts] = new_data
        ts = ts.tz_convert(None)
        #print(dataPoints[symbol])
        if((ts - market_open).seconds > 60 and len(dataPoints[symbol]) > 60):
            # check for a positive ema
            #print(dataPoints[symbol].iloc[-42:])
            hist = ema(20, dataPoints[symbol].iloc[-60:]['close']).diff()
            #print(hist.iloc[-20:])
            bullonly = False
            # Check to see if the last ema point was positive and if the rolling ema is also positive
            if (all(x > 0 for x in hist.iloc[-20:])):
                bullonly = True
                curMod = BullHMM
                print('bull moment')
                pass
            
            elif (all(x < 0 for x in hist.iloc[-20:])):
                # change number to average
                bullonly = False
                curMod = BullHMM
                pass

            else:
                bullonly = False
                return
            # DO CHECKS TO DETERMINE IF WE WILL BE PURCHASING SHARES

            # Stock has passed all checks; figure out how much to buy
            # stop_price = find_stop(
            #     data.close, data[symbol], ts
            # )
            
            
            # stop_prices[symbol] = stop_price
            # target_prices[symbol] = data.close + (
            #     (data.close - stop_price) * 3
            # )
            # print(data.close, stop_price, risk, portfolio_value, target_prices[symbol])
            #print(roll[symbol])
            if(list(roll[symbol].columns) == []):
                roll[symbol] = getHistorical(43*sum(dataPoints[symbol].iloc[-20:]['volume'])/20/6.58345225885229, dataPoints[symbol][-20:])
            else:
                try:
                    current = roll[data.symbol].iloc[-1]
                except KeyError:
                    current = None
                new_data = []
                if current is not None:
                    new_data = [
                        current['open'],
                        data.close,
                        data.low if data.low < current['low'] else current['low'],
                        data.high if data.high > current['high'] else current['high'],
                        current['volume'] + np.log(data.volume)
                    ]
                    print('new data is being acquired correctly')
                else: 
                    new_data = [
                        data.open,
                        data.close,
                        data.low,
                        data.high,
                        np.log(data.volume)
                    ]
                roll[symbol].iloc[-1] = new_data
                print('new data')
            #print(roll[symbol])

            position = pos.get(symbol, 0)
            print(position)
            print(symbol + " = " + str(roll[symbol].iloc[-1]['volume']))
            print(roll[symbol].iloc[-1]['volume'] - 43*sum(dataPoints[symbol][-20:]['volume'])/20/6.588001176795564)
            if(roll[symbol].iloc[-1]['volume'] >= (43*sum(dataPoints[symbol][-20:]['volume'])/20/6.588001176795564)):
                df = roll[symbol].iloc[-1]['volume'] - 43*sum(dataPoints[symbol][-20:]['volume'])/20/6.588001176795564

                # Not actually updating
                roll[symbol].iloc[-1].at['volume'] = roll[symbol].iloc[-1]['volume'] - df
                
                #because a new period has been entered/exited, perform order operation
                shares_to_buy = pfv * .01 // (data.close)
                if shares_to_buy == 0:
                    shares_to_buy = 1
                shares_to_buy -= pos.get(symbol, 0)
                if shares_to_buy <= 0:
                    return

                print(roll[symbol])
                print('df = '+ str(df))

                if(bullonly and predict(curMod, roll[symbol], scale)):

                    existing_order = orders.get(symbol)
                    if existing_order is not None:
                        # Make sure the order's not too old
                        print(existing_order)
                        submission_ts = None
                        try:
                            submission_ts = existing_order.submitted_at.astimezone(None)
                        except e:
                            submission_ts = existing_order['submitted_at'].astimezone(None)
                        
                        
                        order_lifetime = ts - submission_ts
                        if order_lifetime.seconds > 60:
                            # Cancel it so we can try again for a fill
                            api.cancel_order(existing_order.id)
                        return

                    
                    else:
                        print('Submitting buy for {} shares of {} at {}'.format(
                            shares_to_buy, symbol, data.close
                        ))
                        try:
                            o = api.submit_order(
                                symbol=symbol, qty=str(shares_to_buy), side='buy',
                                type='limit', time_in_force='day',
                                limit_price=str(data.close)
                            )
                            orders[symbol] = o

                        except Exception as e:
                            print(e)
                        
                        return

                roll[symbol] = roll[symbol].append([current.open,
                                    current.high,
                                    current.low,
                                    current.close,
                                    df])
                       
                if position > 0:
                    #liquidate all positions
                    print('Submitting sell for {} shares of {} at {}'.format(
                        position, symbol, data.close
                    ))
                    try:
                        o = api.submit_order(
                            symbol=symbol, qty=str(position), side='sell',
                            type='market', time_in_force='day'
                        )
                        orders[symbol] = o

                    except Exception as e:
                        print(e)
                    return
                    
            # elif position > 0:
            #         #liquidate all positions
            #         print('Submitting sell for {} shares of {} at {}'.format(
            #             position, symbol, data.close
            #         ))
            #         try:
            #             o = api.submit_order(
            #                 symbol=symbol, qty=str(position), side='sell',
            #                 type='market', time_in_force='day'
            #             )
            #             orders[symbol] = o

            #         except Exception as e:
            #             print(e)
            #         return

        
        
    
    # Minute-based data.
    @conn.on('AM.*')
    async def handle_minute_bar(conn, channel, data):
        ts = data.start
        symbol = data.symbol
        if(not api.get_clock().is_open):
            symbols.remove(symbol)
            if len(symbols) <= 0:
                conn.close()
                start()
            conn.deregister([
            'AM.{}'.format(symbol),
            'A.{}'.format(symbol)])
   
    #If live trading, make sure to add 'trade_updates' to the beginning of channels
    #and implement the 'trade_updates' websocket.
    channels = ['trade_updates']
    for s in symbols:
        # print(s.ticker)
        symbol_channels = ['A.{}'.format(s.ticker), 'AM.{}'.format(s.ticker)]
        channels += symbol_channels
    print("watching {} symbols".format(len(symbols)))
    #print(channels)
    run_ws(conn, channels)


def run_ws(conn, channels):
    try:
        conn.run(channels)
    except Exception as e:
        print(e)
        conn.close
        run_ws(conn, channels)

def convert(hist):
    #print("Converting data")
    conv = []

    o = np.array(hist['open'])
    c = np.array(hist['close'])
    h = np.array(hist['high'])
    l = np.array(hist['low'])
    
    fracC = []
    fracH = []
    fracL = []

    
    for i in range(len(o.tolist()) if isinstance(o, list) else 1):
    
        if(c[i]-o[i] < 0):
            if((o[i]-c[i])/o[i] >= 1 and (o[i]-c[i])/o[i] <=1.5):
                fracC.append(-.75)
            elif((o[i]-c[i])/o[i] > 1.5):
                fracC.append(-1) 
            else:
                fracC.append(1/np.log((o[i]-c[i])/o[i]))
        elif(c[i]-o[i] > 0):
            if((c[i]-o[i])/o[i] >= 1 and (c[i]-o[i])/o[i] <= 1.5):
                fracC.append(.75)
            elif((c[i]-o[i])/o[i] > 1.5):
                fracC.append(1)
            else:
                fracC.append(-1/np.log((c[i]-o[i])/o[i]))
        else:
            fracC.append(0)

        #upward movements are unbound. should consider a way to account for this.
        if((h[i]-o[i]) <= 0):
            fracH.append(0)
        elif(np.log((h[i]-o[i])/o[i]) >= 0):
            fracH.append(10)
        else:
            fracH.append(-1/np.log((h[i]-o[i])/o[i]))
       
        #l is bound by zero
        if((o[i]-l[i]) <= 0):
            fracL.append(0)
        elif(np.log((o[i]-l[i])/o[i]) == 0):
            fracL.append(10)
        else:
            fracL.append(-1/np.log((o[i]-l[i])/o[i]))


    return np.column_stack((fracC, fracH, fracL))

def predict(hmm, histT, scalar):
    prevD = histT
    #print(prevD)
    conv = convert(prevD)
    conv = np.column_stack(((scalar[0].transform(np.array(conv[:,0]).reshape(-1, 1)).flatten()-.5), (scalar[1].transform(np.array(conv[:,1]).reshape(-1, 1)).flatten()-.5), (scalar[2].transform(np.array(conv[:,2]).reshape(-1, 1)).flatten()-.5)))
    stateSeq = hmm.predict(conv)

    randstate = check_random_state(hmm.random_state)
    nextState = (np.cumsum(hmm.transmat_, axis=1)[stateSeq[-1]] > randstate.rand())
    nextObs = hmm._generate_sample_from_state(nextState.argmax(),randstate)
    return (nextObs[0]>0)


def get_tickers():
    print('Getting current ticker data...')
    tickers = api.polygon.all_tickers()
    print('Success.')
    assets = api.list_assets()
    symbols = [asset.symbol for asset in assets if asset.tradable]
    symbols = ['AMD'] #, 'SPY', 'WM', 'TWTR', 'SQ', 'KO', 'DIS', 'MSFT']
    return [ticker for ticker in tickers if (
        ticker.ticker in symbols and
        ticker.lastTrade['p'] >= 5 and
        ticker.lastTrade['p'] <= 500 and
        ticker.prevDay['v'] > 1000000 #and
        #ticker.todaysChangePerc >= 2.8
    )]



def start():
    nyc = timezone('America/New_York')
    t = datetime.now()
    t.time
    t = t + timedelta(hours=1)
    today = datetime.today()
    today_str = t.strftime('%Y-%m-%d')
    #today_str = datetime.today().astimezone(nyc).strftime('%Y-%m-%d')
    calendar = api.get_calendar(start=today_str, end=today_str)[0]
    market_open = today.replace(
        hour=calendar.open.hour,
        minute=calendar.open.minute,
        second=0
    )
    #market_open = market_open.astimezone(nyc)  
    print(market_open)
    market_close = today.replace(
        hour=calendar.close.hour,
        minute=calendar.close.minute,
        second=0
    )
    #market_close = market_close.astimezone(nyc)
    print(market_close)
    # Wait until just before we might want to trade
    #current_dt = datetime.today() + timedelta(hours=1)
    current_dt = datetime.today() + timedelta(hours=1)
    since_market_open = current_dt - market_open
    print(since_market_open.seconds//60//60)
    while since_market_open.seconds // 60 // 60 >= 7:
        print(since_market_open.seconds//60//60)
        #print(since_market_open.seconds//60//60)
        time.sleep(1)

        current_dt = datetime.today() + timedelta(hours=1)
        since_market_open = current_dt - market_open
    
    run(get_tickers(), market_open, market_close)
    pass


def tickers(period):
    # return stocks symbols with volume above 
    ticks = api.polygon.all_tickers()
    # If you want tradable assets, get api.list_assets() and filter the ones that are .tradable
    # Since I'm not trading and making an indicator, I won't be needing this
    #Given how much volume you want in a period, make sure the stock can see 10 periods in a day. 
    return [t for t in ticks if (t.prevDay['v'] > 10*period)]


if __name__ == "__main__":
    start()