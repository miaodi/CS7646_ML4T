import datetime as dt
import os
import numpy as np
import pandas as pd
from util import get_data, plot_data
import matplotlib.pyplot as plt
import marketsimcode
import indicators


class ManualStrategy:
    def __init__(self) -> None:

        self.BB_window = 20
        self.BB_m = 2

        self.SMA_fast = 20
        self.SMA_slow = 50

        self.RSI_window = 14
        self.RSI_limit = 30
        self.SO_window = 14
        self.SO_K = 3
        self.SO_D = 3

    def setBB(self, window, m):
        self.BB_window = int(window)
        self.BB_m = m

    def setSMA(self, fast, slow):
        self.SMA_fast = int(fast)
        self.SMA_slow = int(slow)

    def setRSI(self, window, limit):
        self.RSI_window = int(window)
        self.RSI_limit = limit

    def setSO(self, window, K, D):
        self.SO_window = int(window)
        self.SO_K = int(K)
        self.SO_D = int(D)

    def setWeights(self, weights):
        self.Weights = weights

    def testPolicy(
        self,
        symbol="IBM",
        sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009, 12, 31),
        sv=10000,
    ):
        self.tickName = symbol
        self.start = sd
        self.end = ed
        self.start_val = sv

        self.initialize()
        return self.Signal()

    def initialize(self):
        dates = pd.date_range(self.start - 80*pd.Timedelta(days=1), self.end)
    # print(dates)
        self.prices = get_data([self.tickName], dates, colname="Adj Close").drop(
            columns=["SPY"])  # automatically adds SPY
        self.prices['Volume'] = get_data([self.tickName], dates, colname="Volume").drop(
            columns=["SPY"])  # automatically adds SPY

        # create signals
        self.SMASignal(self.SMA_fast, self.SMA_slow)
        self.BollingerBandsSignal(self.BB_window, self.BB_m)
        self.RSISignal(self.RSI_window, self.RSI_limit)
        self.StochasticOscillatorSignal(self.SO_window, self.SO_K, self.SO_D)

        self.prices = self.prices[self.start:self.end]

    def SMASignal(self, fast=10, slow=30):
        fastName = indicators.SMA(self.prices, self.tickName, fast)
        slowName = indicators.SMA(self.prices, self.tickName, slow)
        self.CrossSignal('SMA', fastName, slowName)

    def BollingerBandsSignal(self, window=20, m=2):
        [name, low, high] = indicators.BollingerBands(
            self.prices, self.tickName, window, m)

        signalName = 'BB-Signal'
        self.prices[signalName] = np.where(
            self.prices[self.tickName] < self.prices[low], 1, 0) + np.where(
            self.prices[self.tickName] > self.prices[high], -1, 0)

    def RSISignal(self, window=14, limit=30):
        name = indicators.RSI(self.prices, self.tickName, window)

        signalName = 'RSI-Signal'
        self.prices[signalName] = np.where(
            self.prices['RSI-'+str(window)] < limit, 1, 0) + np.where(
            self.prices['RSI-'+str(window)] > 100-limit, -1, 0)

    def StochasticOscillatorSignal(self, window=14, K=3, D=3):
        [fastName, slowName] = indicators.StochasticOscillator(
            self.prices, self.tickName, window, K, D)
        self.CrossSignal('SO', fastName, slowName)

        # cross and slow goes below fast => buy signal (1)
        # cross and slow goes above fast => sell signal (-1)
        # return name-Signal in prices

    def CrossSignal(self, name, fastName, slowName):
        signalName = name+'-Signal'
        self.prices[signalName] = np.sign(
            self.prices[fastName] - self.prices[slowName])
        self.prices[signalName] = self.prices[signalName].diff()/2

    def Signal(self):
        weightSum = np.sum(self.Weights)
        weightedSum = self.prices['SMA-Signal']*self.Weights[0] + \
            self.prices['BB-Signal']*self.Weights[1]+self.prices['RSI-Signal'] * \
            self.Weights[2]+self.prices['SO-Signal']*self.Weights[3]
        weightedSum.div(weightSum)
        self.signal = pd.DataFrame(data=np.where(weightedSum < -1./3, -1, 0) +
                                   np.where(weightedSum > 1./3, 1, 0), index=self.prices.index, columns={'signal'})
        prev = 0
        for i in range(0, len(self.signal)):
            if self.signal.iloc[i][0] != 0 and self.signal.iloc[i][0] != prev:
                prev = self.signal.iloc[i][0]
            else:
                self.signal.iloc[i][0] = 0
        return self.signal

    def portValue(self, commission=9.95, impact=0.005, stock_limit=1000):
        stock = 0
        start_val = self.start_val
        res = pd.DataFrame(np.zeros(
            shape=(self.prices.shape[0], 1)), index=self.prices.index, columns={'value'})
        for i in range(0, len(self.prices)):
            stock_price = self.prices.iloc[i][self.tickName]
            if self.signal.iloc[i][0] == 0:
                res.iloc[i][0] = start_val + stock * stock_price
                continue
            if self.signal.iloc[i][0] == 1:
                adjusted_stock_price = stock_price*(1+impact)
                target = stock_limit
            elif self.signal.iloc[i][0] == -1:
                adjusted_stock_price = stock_price*(1-impact)
                target = -stock_limit
            if target != stock:
                start_val += -adjusted_stock_price*(target-stock)-commission
                stock = target
            res.iloc[i][0] = start_val+stock*stock_price
        res = res/(res.iloc[0])
        return res


def BenchMark(
        symbol="IBM",
        sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009, 12, 31),
        sv=10000):
    dates = pd.date_range(sd, ed)
    prices = 1000*get_data([symbol], dates,
                           addSPY=True, colname="Adj Close").drop(columns="SPY")
    rest = sv-prices.iloc[0][symbol]
    benchmark = prices[symbol]+rest
    benchmark = prices[symbol] / prices.iloc[0][symbol]
    return benchmark


def optimize():
    benchMark = BenchMark('JPM', dt.datetime(2008, 1, 1),
                          dt.datetime(2009, 12, 31), 100000)
    ms = ManualStrategy()
    # ms.setSMA(10, 34)
    # ms.setRSI(13, 19)
    # ms.setSO(20, 6, 1)
    # ms.setBB(16, 1.47924663)
    # ms.setWeights([1, 1, 1, 1])
    # ms.setWeights([0.76029473, 0.79502203, 0.93583619, 0.46552834])

    # optimize SMA
    ms.setWeights([1, 0, 0, 0])
    # default
    ms.setSMA(20, 50)
    ms.testPolicy('JPM', dt.datetime(2008, 1, 1),
                  dt.datetime(2009, 12, 31), 100000)
    defaultRes = ms.portValue()

    # optimized
    ms.setSMA(10, 34)
    ms.testPolicy('JPM', dt.datetime(2008, 1, 1),
                  dt.datetime(2009, 12, 31), 100000)
    optRes = ms.portValue()
    fig, (ax1) = plt.subplots(figsize=(10, 6))

    benchMark.plot(ax=ax1, color='purple')
    defaultRes.plot(ax=ax1, color='green')
    optRes.plot(ax=ax1, color='red')
    fig.suptitle('SMA indicator')
    plt.legend(['Benchmark', 'Default', 'Optimized'], loc='upper left')
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Normalized value")
    plt.savefig(
        "./strategy_evaluation_2023Sum/strategy_evaluation/plots/SMA_opt.eps", format='eps')

    # optimize BB
    ms.setWeights([0, 1, 0, 0])
    # default
    ms.setBB(20, 2)
    ms.testPolicy('JPM', dt.datetime(2008, 1, 1),
                  dt.datetime(2009, 12, 31), 100000)
    defaultRes = ms.portValue()

    # optimized
    ms.setBB(16, 1.47924663)
    ms.testPolicy('JPM', dt.datetime(2008, 1, 1),
                  dt.datetime(2009, 12, 31), 100000)
    optRes = ms.portValue()
    fig, (ax1) = plt.subplots(figsize=(10, 6))

    benchMark.plot(ax=ax1, color='purple')
    defaultRes.plot(ax=ax1, color='green')
    optRes.plot(ax=ax1, color='red')
    fig.suptitle('Bollinger-Bands indicator')
    plt.legend(['Benchmark', 'Default', 'Optimized'], loc='upper left')
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Normalized value")
    plt.savefig(
        "./strategy_evaluation_2023Sum/strategy_evaluation/plots/BB_opt.eps", format='eps')

    # optimize RSI
    ms.setWeights([0, 0, 1, 0])
    # default
    ms.setRSI(14, 30)
    ms.testPolicy('JPM', dt.datetime(2008, 1, 1),
                  dt.datetime(2009, 12, 31), 100000)
    defaultRes = ms.portValue()

    # optimized
    ms.setRSI(13, 19)
    ms.testPolicy('JPM', dt.datetime(2008, 1, 1),
                  dt.datetime(2009, 12, 31), 100000)
    optRes = ms.portValue()
    fig, (ax1) = plt.subplots(figsize=(10, 6))

    benchMark.plot(ax=ax1, color='purple')
    defaultRes.plot(ax=ax1, color='green')
    optRes.plot(ax=ax1, color='red')
    fig.suptitle('RSI indicator')
    plt.legend(['Benchmark', 'Default', 'Optimized'], loc='upper left')
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Normalized value")
    plt.savefig(
        "./strategy_evaluation_2023Sum/strategy_evaluation/plots/RSI.eps", format='eps')

    # optimize SO
    ms.setWeights([0, 0, 0, 1])
    # default
    ms.setSO(14, 3, 3)
    ms.testPolicy('JPM', dt.datetime(2008, 1, 1),
                  dt.datetime(2009, 12, 31), 100000)
    defaultRes = ms.portValue()

    # optimized
    ms.setSO(20, 6, 1)
    ms.testPolicy('JPM', dt.datetime(2008, 1, 1),
                  dt.datetime(2009, 12, 31), 100000)
    optRes = ms.portValue()
    fig, (ax1) = plt.subplots(figsize=(10, 6))

    benchMark.plot(ax=ax1, color='purple')
    defaultRes.plot(ax=ax1, color='green')
    optRes.plot(ax=ax1, color='red')
    fig.suptitle('Full Stochastic Oscillator')
    plt.legend(['Benchmark', 'Default', 'Optimized'], loc='upper left')
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Normalized value")
    plt.savefig(
        "./strategy_evaluation_2023Sum/strategy_evaluation/plots/SO.eps", format='eps')

    # optimize weights
    ms.setSMA(10, 34)
    ms.setRSI(13, 19)
    ms.setSO(20, 6, 1)
    ms.setBB(16, 1.47924663)
    # default
    ms.setWeights([1, 1, 1, 1])
    ms.testPolicy('JPM', dt.datetime(2008, 1, 1),
                  dt.datetime(2009, 12, 31), 100000)
    defaultRes = ms.portValue()

    # optimized
    ms.setWeights([0.76029473, 0.79502203, 0.93583619, 0.46552834])
    ms.testPolicy('JPM', dt.datetime(2008, 1, 1),
                  dt.datetime(2009, 12, 31), 100000)
    optRes = ms.portValue()
    fig, (ax1) = plt.subplots(figsize=(10, 6))

    benchMark.plot(ax=ax1, color='purple')
    defaultRes.plot(ax=ax1, color='green')
    optRes.plot(ax=ax1, color='red')
    fig.suptitle('Manual Strategy')
    plt.legend(['Benchmark', 'Default', 'Optimized'], loc='upper left')
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Normalized value")
    plt.savefig(
        "./strategy_evaluation_2023Sum/strategy_evaluation/plots/Weight.eps", format='eps')


def InSample(
        sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009, 12, 31),
        filename='ManualInSample.eps',
        insample=True):
    benchMark = BenchMark('JPM', sd, ed, 100000)
    ms = ManualStrategy()
    # Manual Strategy
    ms.setSMA(10, 34)
    ms.setRSI(13, 19)
    ms.setSO(20, 6, 1)
    ms.setBB(16, 1.47924663)

    # optimized
    ms.setWeights([0.76029473, 0.79502203, 0.93583619, 0.46552834])
    ms.testPolicy('JPM', sd, ed, 100000)
    signal = ms.signal
    # print(signal.iloc[0].name)
    optRes = ms.portValue()
    fig, (ax1) = plt.subplots(figsize=(10, 6))
    ymin, ymax = ax1.get_ylim()
    benchMark.plot(ax=ax1, color='purple')
    optRes.plot(ax=ax1, color='red')
    for i in range(0, len(signal)):
        if signal.iloc[i][0] == 1:
            ax1.axvline(x=signal.iloc[i].name, color='blue')
        elif signal.iloc[i][0] == -1:
            ax1.axvline(x=signal.iloc[i].name, color='black')
    if insample:
        fig.suptitle('In sample Manual Strategy vs Benchmark')
    else:
        fig.suptitle('Out of sample Manual Strategy vs Benchmark')
    ax1.legend(['Benchmark', 'Manual Strategy'], loc='upper left')
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Normalized value")
    plt.savefig(
        "./strategy_evaluation_2023Sum/strategy_evaluation/plots/"+filename, format='eps')


def OutOfSample():
    InSample(sd=dt.datetime(2010, 1, 1),
             ed=dt.datetime(2011, 12, 31),
             filename='ManualOutOfSample.eps', insample=False)
