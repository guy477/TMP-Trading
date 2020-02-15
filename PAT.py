import requests, time, json
import pandas as pd
import numpy as np
#import cupy as np
import numba as nb
from collections import *
import itertools
import matplotlib.pyplot as pl
from hmmlearn import hmm
import random
from multiprocessing import Pool
from threading import Thread
from sklearn.utils import check_random_state
import time
from tqdm import tqdm



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
#cred = json.load(open("credentials.json"))
files = json.load(open("files.json"))

def readFiles():
    data0 = pd.read_csv(''+files['ETH']+'/2016/merged.csv')
    data0.columns = ['time', 'open', 'close', 'high', 'min', 'volume']
    data1 = pd.read_csv(''+files['ETH']+'/2017/merged.csv')
    data1.columns = ['time', 'open', 'close', 'high', 'min', 'volume']
    data2 = pd.read_csv(''+files['ETH']+'/2018/merged.csv')
    data2.columns = ['time', 'open', 'close', 'high', 'min', 'volume']
    return pd.concat([data0, data1, data2], ignore_index=True)

def readTestFiles():
    data = pd.read_csv(''+files['ETH']+'/2019/merged.csv')
    data.columns = ['time', 'open', 'close', 'high', 'min', 'volume']
    return pd.concat([data], ignore_index=True)

def getHistorical(period, datas):
    hist = []
    conv = []
    o = -1
    
    #print("Aggregating historical data")
    data = datas 
    #print(data.head())

    #      open, close, min, max
    vals = [o, -1, float('inf'), -1]
    #convV = [0, 0, 0]
    s = 0

    for ind, p in data.iterrows():
        # get the open, close, min, and max for each volume period given the minute based data.
        if(vals[0] == -1):
            vals[0] = p['open']
        vals = [vals[0], -1, min(vals[2], p['min']), max(vals[3], p['high'])]
        s += p['volume']
        if (s >= period):
            dif = period - s
            vals[1] = p['close']
            hist.append(vals)
            if(dif!=0):
                o = p['close']
            else:
                o = -1
            vals = [o, -1, float('inf'), -1]
            s=0
    hist = pd.DataFrame(hist, columns = ['open', 'close', 'min', 'max'])
    return hist#, conv


def getHistoricalTest(period, datas):
    hist = []
    conv = []
    data = datas
    o=-1
    vals = [o, -1, float('inf'), -1]
    #convV = [0, 0, 0]
    s = 0

    for ind, p in data.iterrows():
        # get the open, close, min, and max for each volume period given the minute based data.
        if(vals[0] == -1):
            vals[0] = p['open']
        vals = [vals[0], -1, min(vals[2], p['min']), max(vals[3], p['high'])]
        s += p['volume']
        if (s >= period):
            dif = period - s 
            vals[1] = p['close']
            hist.append(vals)
            if(dif!=0):
                o = p['close']
            else:
                o = -1
            vals = [-1, -1, float('inf'), -1]
            s=0
    hist = pd.DataFrame(hist, columns = ['open', 'close', 'min', 'max'])
    return hist#, conv


"""
    converts from open, close, min, max
    to period change, max/min, fractional high, and fractional low
"""
def convert(hist):
    #print("Converting data")
    conv = []

    o = np.array(hist['open'])
    c = np.array(hist['close'])
    h = np.array(hist['max'])
    l = np.array(hist['min'])
    
    return np.column_stack((np.log(1+(c-o)/o),np.log(1 + (h-o)/o) ,np.log(1 + (o-l)/o)))
        


def run(period):
    print('getting historical')
    hist = getHistorical(period, readFiles())
    print('getting historical test')
    histT = getHistoricalTest(period, readTestFiles())
    
    conv = convert(hist)

    hist.to_csv('hist.csv')
    histT.to_csv('histT.csv')
    pd.DataFrame(conv).to_csv('converted.csv')
    

    for i in conv:
        print(i)
    
#-------------------------------------------------------------------------------------------------------------------

    print('make hmm')
    
    HMM = hmm.GaussianHMM(n_components = 7 , covariance_type="full", random_state=7, n_iter = 1000)

    HMM.fit(conv)
    print(HMM.sample(10))
    print(HMM.transmat_)
    print('complete')
    
#-------------------------------------------------------------------------------------------------------------------
    scores  = defaultdict(list)
    strt = random.randint(5, histT.__len__()-75)
    for j in range(15):
        pSize = random.randint(10, 150)
        
        
        for i in range(50):
            #if(i == 0 and not scores[pSize] == None):
            #    break
            strt = random.randint(5, histT.__len__()-pSize)
            pred, sc = predict(HMM, histT, strt, strt+pSize, 4, False)
            scores[pSize].append(sc)
        

#-------------------------------------------------------------------------------------------------------------------

    predictedCloseForTest, _ = predict(HMM, histT, strt, strt+75, 4, True)
    trueOpenForTest       = histT.iloc[strt:strt+75]['open'].values
    trueCloseForTest      = histT.iloc[strt:strt+75]['close'].values

    print("50 random periods w/50 different random tests resuts::")

    for i in scores.keys():
        s = str(sum(scores[i])/len(scores[i]))[0:5]
        print("For the 50 random tests over " + str(i) + " periods, the HMM determined the direction correctly: " + s + "% of the time.")
        #plotter(trueCloseForTest, predictedCloseForTest, trueOpenForTest, )
    

#scoresDebugg = defaultdict(int)

def optimize():
    # Dictionary from Period to dict from HMM components to dict from HMM lookBack size to list of tuples of test length and score
    optimizer = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

    # Volums will be three years worth of data
    vol = int(readFiles()['volume'].sum())
    #print(vol)

    averageBest50 = []
    

    # min, the data will, on average, make up an hour
    # max, the data will, on average, make up three days
    # increment by the hour average::: total iterations:::: 36

    for i in tqdm(range(vol//1095//24, vol//365, vol//1095//12), desc="Volume Progress"):
        hist = getHistorical(i, readFiles())
        histT = getHistoricalTest(i, readTestFiles())
        conv = convert(hist)
        #check components
        for j in tqdm(range(3, 10), desc="Components Progress"):
            #check look-backs
            HMM = hmm.GaussianHMM(n_components = j , covariance_type="full", random_state=7, n_iter = 1000)
            HMM.fit(conv)
            res = []
            with Pool() as p:

                res = p.starmap(runTests, [(HMM, histT, 4, 50, x) for x in range(1, 9)])
            
            optimizer[i][j][0] = res[0]
            optimizer[i][j][1] = res[1]
            optimizer[i][j][2] = res[2]
            optimizer[i][j][3] = res[3]
            optimizer[i][j][4] = res[4]
            optimizer[i][j][5] = res[5]
            optimizer[i][j][6] = res[6]
            optimizer[i][j][7] = res[7]
            #for k in tqdm(range(1, 8, 4), desc="Look-back Progress"):
                
                
        for j in optimizer[i].keys():
            for k in optimizer[i][j].keys():
                s = 0
                for l in optimizer[i][j][k].keys():
                    s += sum(optimizer[i][j][k][l])/len(optimizer[i][j][k][l])

                sc = s/len(optimizer[i][j][k].keys())

                if len(averageBest50) == 0 or averageBest50[-1][3] < sc:
                    averageBest50.append((i, j, k, sc))
                    averageBest50.sort(key = lambda x: x[3], reverse=True)
                    if len(averageBest50) > 50:
                        averageBest50.pop()
        print(averageBest50[0:8])
        

    return averageBest50

def runTests(HMM, histT, iter1, iter2, lookBack):
    scores  = defaultdict(list)
    strt = 0
    for j in range(iter1):
        pSize = random.randint(10, 150)
        for i in range(iter2):
            strt = random.randint(lookBack+1, histT.__len__()-pSize)
            pred, sc = predict(HMM, histT, strt, strt+pSize, lookBack, False)
            scores[pSize].append(sc)
    return scores


def predict(hmm, histT, startInd, endInd, lookBack, plot):
    pred = []

    for i in range(startInd, endInd):
        oPrice = histT.iloc[i]['open']
        cPrice = histT.iloc[i]['close']

        prevD = histT.iloc[i-lookBack:i]
        
        conv = convert(prevD)
        
        stateSeq = hmm.predict(conv)
        randstate = check_random_state(hmm.random_state)
        nextState = (np.cumsum(hmm.transmat_, axis=1)[stateSeq[-1]] > randstate.rand()).argmax()
        nextObs = hmm._generate_sample_from_state(nextState,randstate)
        pred.append(oPrice * (1+nextObs[0]))

    c = 0
    s = 0
    for i in histT.iloc[startInd:endInd]['open'].values:
        if((histT.iloc[s+startInd]['close']-i)*(pred[s]-i)>=0):
            c+=1
        s+=1
    #print("for this sample, the HMM predicted the correct direction " + str(100*(c/s)) + "% of the time. P = " + str(endInd-startInd) + ".")
    
    if(plot):
        #plotter(histT.iloc[startInd:endInd]['close'].values)
        plotter(histT.iloc[startInd:endInd]['close'].values, pred, histT.iloc[startInd:endInd]['open'].values, ""+str(endInd - startInd)+"-"+str(100*(c/s))[0:4])

    return pred, (100*(c/s))



# The data we get will be all over the place. this will "standardize" the datapoints we see.
# This can greatly increase the predict functions overall speed.
# POTENTIAL OPTIMIZATION SPEED UP SOLUTION.
def possibleDataVals(fracS, fracHS, fracLS):
    print("Defining possibleoutcomes")
    return np.array(list(itertools.product(np.linspace(-.1, .1, fracS), np.linspace(0, .1, fracHS), np.linspace(0, .1, fracLS))))

def plotter(data, dataP, dataPO, name):
    #print(data)
    #print(dataP)
    pl.style.use('ggplot')
    plot = pl.figure()
    axis = plot.add_subplot(111)
    axis.plot([x for x in range(data.__len__())], data, 'bo-', label='real close')
    axis.plot([x for x in range(data.__len__())], dataP, 'r+-', label='predicted close (based on realO)')
    axis.plot([x for x in range(data.__len__())], dataPO, 'b+-', label='real open')
    pl.legend()
    pl.savefig("plots/"+name+".png")
    pl.close(plot)

def start():
    #t = api.get_clock()

    # PERIODS::: (average 12 ticks a day)
    #   XRP: 1934152
    #   XRP: ‭23209824‬ (average of a day)
    #   ETH: 250000   (works really well.)

    period = int(input("INPUT VOLUME PERIOD SIZE: "))
    run(period)
    # CRYTO CURRENTLY 
    #if t.is_open == False:
    #        tillopen = (t.next_open - t.timestamp).total_seconds()
    #        print("market closed. Sleep for ", int(tillopen)-60, " seconds")
    #        time.sleep(int(tillopen))
    


"""

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
"""


if __name__ == "__main__":
    optimize()
    start()
    