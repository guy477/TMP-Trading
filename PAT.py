import alpaca_trade_api as tradeapi
import requests, time, json
#import ta.volatility as ta
import pandas as pd



"""
 *  This program will be a little different from the last.
 *  Instead of sticking with normal timeseries data, I will
 *  use volume based data. What I mean by this is instead of
 *  having periods be completed after certain periods of time,
 *  periods will end after a certian amount of volume has been
 *  accumulated.
 *  
 *  The idealistic benefit of making this change is that
 *  the data will better represent 'true' price movements.
 *  Price movement isn't driven by time, price movement is
 *  driven by volume. To see this, time can move forward with
 *  no volume and the price will never change because of
 *  the lack of voulme.
 *
 *  Using volume will add challenges from a programming standpoints.
 *  Where in timeseries data, periods end regularly and are determined
 *  external from the market, with volume based data, trades aren't
 *  and depending on the market being conscious of these periods sizes 
 *  will be of much greater importance. 
 *  Along with this, visualizing will be very important for me to make
 *  sense of the data I'm seeing. I've been finding it difficult to
 *  locate anything on this form of analysis.

 Number of bullish/bearish periods and their ratio
 Body size of bullish/bearish periods
 Number of consecutive periods
"""



#Load credentials from json
cred = json.load(open("credentials.json"))
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

def getHistorical(symbols, period):
    hist = pd.DataFrame()

    for s in symbols:
        # Get the last 3 days worth of data in minute form. If you can
        # the smaller the timeframe the better the data will be.
        minH = api.polygon.historic_agg(size='minute', symbol=s, limit=1260).df
        min = minH['volume']
        s = 0
        c = 0
        for v in minH['volume']:
            # get the min, max, open, and close for each period
            s += v
            if (s >= period):
                hist.insert(20000*c, 'min')





def run(tickers, period):
       
    # stream connection
    conn = tradeapi.StreamConn(base_url=base_url, key_id=api_key_id, secret_key=api_secret)

    # Update initial state with information from tickers
    
    symbols = tickers

    #get historical minute data to populate df.
    minHist = getHistorical(symbols, period)

    print('running')
    
    
    #@conn.on('trade_update')
    #async def handle_trade_update(conn, channel, data):
    #    print("handle portfolio changes if live trading")

    # Second-based data.
    @conn.on('A.*')
    async def handle_second_bar(conn, channel, data):
        symbol = data.symbol
        

        if(not api.get_clock().is_open):
            symbols.remove(symbol)
            if len(symbols) <= 0:
                conn.close()
                start()
            conn.deregister([
            'AM.{}'.format(symbol),
            'A.{}'.format(symbol)])
        

    

    # Minute-based data.
    @conn.on('AM.*')
    async def handle_minute_bar(conn, channel, data):
        ts = data.start

   
    #If live trading, make sure to add 'trade_updates' to the beginning of channels
    #and implement the 'trade_updates' websocket.
    channels = []
    for s in symbols:
        print(s.ticker)
        symbol_channels = ['A.{}'.format(s.ticker), 'AM.{}'.format(s.ticker)]
        channels += symbol_channels
    print("watching {} symbols".format(len(symbols)))
    print(channels)
    run_ws(conn, channels)


def tickers(period):
    # return stocks symbols with volume above 
    ticks = api.polygon.all_tickers()
    # If you want tradable assets, get api.list_assets() and filter the ones that are .tradable
    # Since I'm not trading and making an indicator, I won't be needing this

    #Given how much volume you want in a period, make sure the stock can see 10 periods in a day. 
    return [t for t in tick if (t.prevDay['v'] > 10*period)]

def plotter(data):
       

def start():
    t = api.get_clock()
    period = int(input("INPUT VOLUME PERIOD SIZE"))

    if t.is_open == False:
            tillopen = (t.next_open - t.timestamp).total_seconds()
            print("market closed. Sleep for ", int(tillopen)-60, " seconds")
            time.sleep(int(tillopen))
        

    run(tickers(), period)

if __name__ == "__main__":
    start()
    