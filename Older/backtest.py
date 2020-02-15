import alpaca_trade_api
import alpaca_backtrader_api as aba
import numpy
import json
import pandas as pd
import ta
from datetime import datetime
import backtrader as bt
from backtrader.indicators import ExponentialMovingAverage as EMA
from backtrader.indicators import AverageTrueRange as ATR
from backtrader import Indicator


"""
 *
 * Simple program to quickly test strategies developed using 
 * backtrader and Alpaca Market's intregration with Polygon.io
 * for data.
 *
"""

#because backtester didnt have keltner channels installed with it
class KeltnerChannel(Indicator):
    '''
    The Keltner Channels (KC) indicator is a banded indicator similar to Bollinger Bands
    and Moving Average Envelopes. They consist of an Upper Envelope above a Middle Line
    as well as a Lower Envelope below the Middle Line.

    Formula:
      - midband = Exponential Moving Average(close, period)
      - topband = Exponential Moving Average + 2*(Average True Range)(data, period)
      - botband = Exponential Moving Average - 2*(Average True Range)(data, period)

    See:
      - https://www.tradingview.com/wiki/Keltner_Channels_(KC)
    '''
    alias = ('KChannel',)

    lines = ('mid', 'top', 'bot',)
    params = (('period', 20), ('devfactor', 2.0), ('EMA', EMA), ('ATR', ATR),)

    plotinfo = dict(subplot=False)
    plotlines = dict(
        mid=dict(ls='--'),
        top=dict(_samecolor=True),
        bot=dict(_samecolor=True),
    )

    def _plotlabel(self):
        plabels = [self.p.period, self.p.devfactor]
        plabels += [self.p.EMA] * self.p.notdefault('EMA')
        return plabels

    def __init__(self):
        self.lines.mid = ma = self.p.EMA(self.data, period=self.p.period)
        self.ATR = atr = self.p.ATR(self.data, period = self.p.period)
        self.lines.top = ma + 2*atr
        self.lines.bot = ma - 2*atr

        super(KeltnerChannel, self).__init__()




class Strategy(bt.SignalStrategy):
    def __init__(self):
        #implement strategy using bt indicators
        bolband = bt.ind.BollingerBands()
        kelchan = KeltnerChannel()
        cross = bt.ind.CrossOver(bolband.bot, kelchan.bot)
        self.signal_add(bt.SIGNAL_LONG, cross)
        pass

def testOnMarket(startdate, enddate, strat, cash, risk, validticks, DataFactory):
    endvals = []
    for tick in validticks:
        cerebro = bt.Cerebro()
        cerebro.addstrategy(strat)
        cerebro.broker.setcash(cash)
        cerebro.broker.setcommission(commission=0.0)
        cerebro.addsizer(bt.sizers.PercentSizer, percents=risk)
        
        data0 = DataFactory(dataname=tick, historical=True, fromdate=pd.Timestamp(startdate), todate=pd.Timestamp(enddate), timeframe=bt.TimeFrame.TFrame("Days"))

        cerebro.adddata(data0)

        cerebro.run()
        cerebro.plot()
        if(cerebro.broker.getvalue()!=cash):
            endvals.append(cerebro.broker.getvalue())
            print(tick, 'final value: %.2f' %cerebro.broker.getvalue())
    
        
    print(sum(endvals)/float(len(endvals)))



cred = json.load(open("credentials.json"))

api_key = cred["key"]
api_secret = cred["secret"]

alpaca = aba.AlpacaStore(key_id = api_key, secret_key = api_secret, paper=True)

data = alpaca.getdata
data0 = data(dataname = 'JPM', historical = True, fromdate=datetime(2018, 8, 1), timeframe=bt.TimeFrame.Days)



testOnMarket(datetime(2019, 8, 1), datetime.today(), Strategy, 500, 15, ['AAPL', 'JPM', 'AMD', 'SNAP', 'CGC', 'TWTR', 'GPRO', 'AMRN'], data)

    