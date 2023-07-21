import datetime as dt
import os
import numpy as np
import pandas as pd
from util import get_data, plot_data
import matplotlib.pyplot as plt
import marketsimcode
import indicators


class ManualStrategy:
    def __init__(self, name, sd, ed) -> None:
        self.tickName = name
        self.start = sd
        self.end = ed

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

    def Signal(self, weights):
        weightSum = np.sum(weights)
        weightedSum = self.prices['SMA-Signal']*weights[0] + \
            self.prices['BB-Signal']*weights[1]+self.prices['RSI-Signal'] * \
            weights[2]+self.prices['SO-Signal']*weights[3]
        weightedSum.div(weightSum)
        self.signal = pd.DataFrame(data=np.where(weightedSum < -1./3, -1, 0) +
                                   np.where(weightedSum > 1./3, 1, 0), index=self.prices.index)
        prev = 0
        for i in range(0, len(self.signal)):
            if self.signal.iloc[i][0] != 0 and self.signal.iloc[i][0] != prev:
                prev = self.signal.iloc[i][0]
            else:
                self.signal.iloc[i][0] = 0
        return self.signal

    def portValue(self, start_val=100000, commission=9.95, impact=0.005, stock_limit=1000):
        stock = 0
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
        return res


# ms = ManualStrategy('JPM', dt.datetime(2010, 1, 1), dt.datetime(2011, 12, 31))
ms = ManualStrategy(
            'JPM', dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31))
# ms.setSMA(10, 34)
# ms.setRSI(13, 19)
# ms.setSO(20, 6, 1)
# ms.setBB(16, 1.47924663)
ms.initialize()

# signal = ms.Signal([0.76029473, 0.79502203, 0.93583619, 0.46552834])
signal = ms.Signal([1, 1, 1, 1])
res = ms.portValue()
prices = ms.prices
fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10, 6))
prices[ms.tickName].plot(ax=ax1)
# prices['SO'].plot(ax=ax1)
# prices['K%'].plot(ax=ax1)
# prices['D%'].plot(ax=ax1)
signal.plot(ax=ax2)
res.plot(ax=ax3)
ax1.set_xticks([])
ax1.set_xlim([ms.start, ms.end])
ax3.set_xlim([ms.start, ms.end])
plt.show()
print(res.tail(1).values[0][0])
