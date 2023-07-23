""""""
"""  		  	   		  		 			  		 			 	 	 		 		 	
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  		 			  		 			 	 	 		 		 	
Atlanta, Georgia 30332  		  	   		  		 			  		 			 	 	 		 		 	
All Rights Reserved  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
Template code for CS 4646/7646  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  		 			  		 			 	 	 		 		 	
works, including solutions to the projects assigned in this course. Students  		  	   		  		 			  		 			 	 	 		 		 	
and other users of this template code are advised not to share it with others  		  	   		  		 			  		 			 	 	 		 		 	
or to make it available on publicly viewable websites including repositories  		  	   		  		 			  		 			 	 	 		 		 	
such as github and gitlab.  This copyright statement should not be removed  		  	   		  		 			  		 			 	 	 		 		 	
or edited.  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
We do grant permission to share solutions privately with non-students such  		  	   		  		 			  		 			 	 	 		 		 	
as potential employers. However, sharing with other current or future  		  	   		  		 			  		 			 	 	 		 		 	
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  		 			  		 			 	 	 		 		 	
GT honor code violation.  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
-----do not edit anything above this line---  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
Student Name: Tucker Balch (replace with your name)  		  	   		  		 			  		 			 	 	 		 		 	
GT User ID: tb34 (replace with your User ID)  		  	   		  		 			  		 			 	 	 		 		 	
GT ID: 900897987 (replace with your GT ID)  		  	   		  		 			  		 			 	 	 		 		 	
"""




import datetime as dt
import random
import pandas as pd
import util as ut
import indicators
import numpy as np
import QLearner
import matplotlib.pyplot as plt
class StrategyLearner(object):
    """  		  	   		  		 			  		 			 	 	 		 		 	
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		  		 			  		 			 	 	 		 		 	

    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		  		 			  		 			 	 	 		 		 	
        If verbose = False your code should not generate ANY output.  		  	   		  		 			  		 			 	 	 		 		 	
    :type verbose: bool  		  	   		  		 			  		 			 	 	 		 		 	
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		  		 			  		 			 	 	 		 		 	
    :type impact: float  		  	   		  		 			  		 			 	 	 		 		 	
    :param commission: The commission amount charged, defaults to 0.0  		  	   		  		 			  		 			 	 	 		 		 	
    :type commission: float  		  	   		  		 			  		 			 	 	 		 		 	
    """
    # constructor

    def __init__(self, verbose=False, impact=0.0, commission=0.0):
        """  		  	   		  		 			  		 			 	 	 		 		 	
        Constructor method  		  	   		  		 			  		 			 	 	 		 		 	
        """
        self.verbose = verbose
        self.impact = impact
        self.commission = commission
        self.BB_window = 20
        self.BB_m = 2

        self.SMA_fast = 20
        self.SMA_slow = 50

        self.RSI_window = 14
        self.RSI_limit = 30
        self.SO_window = 14
        self.SO_K = 3
        self.SO_D = 3
        self.bins = {}
        self.catagories = {}
        self.signals = []

        self.discretize_size = 5
        self.stock_limit = 1000

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

    # this method should create a QLearner, and train it for trading

    def add_evidence(
        self,
        symbol="IBM",
        sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009, 1, 1),
        sv=10000,
    ):
        """  		  	   		  		 			  		 			 	 	 		 		 	
        Trains your strategy learner over a given time frame.  		  	   		  		 			  		 			 	 	 		 		 	

        :param symbol: The stock symbol to train on  		  	   		  		 			  		 			 	 	 		 		 	
        :type symbol: str  		  	   		  		 			  		 			 	 	 		 		 	
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		  		 			  		 			 	 	 		 		 	
        :type sd: datetime  		  	   		  		 			  		 			 	 	 		 		 	
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		  		 			  		 			 	 	 		 		 	
        :type ed: datetime  		  	   		  		 			  		 			 	 	 		 		 	
        :param sv: The starting value of the portfolio  		  	   		  		 			  		 			 	 	 		 		 	
        :type sv: int  		  	   		  		 			  		 			 	 	 		 		 	
        """

        # add your code to do learning here

        # example usage of the old backward compatible util function
        self.start = sd
        self.end = ed
        self.tickName = symbol
        dates = pd.date_range(self.start - 80*pd.Timedelta(days=1), self.end)
        self.prices = ut.get_data([self.tickName], dates, colname="Adj Close").drop(
            columns=["SPY"])

        self.signals.append(self.SMASignal(self.SMA_fast, self.SMA_slow))
        self.signals.append(self.BollingerBandsSignal(
            self.BB_window, self.BB_m))
        self.signals.append(self.RSISignal(self.RSI_window, self.RSI_limit))
        self.signals.append(self.StochasticOscillatorSignal(
            self.SO_window, self.SO_K, self.SO_D))

        self.prices = self.prices[self.start:self.end]
        for i in self.signals:
            self.CreateBin(i)

        # multiplier
        self.multiplier = np.array([1])
        for i in self.signals:
            self.multiplier = np.append(self.multiplier,
                                        self.multiplier[-1]*(len(self.bins[i])-1))

        # default Qlearner
        # action 0: sell | 1: hold | 2: buy
        self.Q = QLearner.QLearner(
            num_states=self.NumOfStates()*self.NumOfStates(), num_actions=3)

        catagoriesMatrix = np.array([self.catagories[i] for i in self.signals])
        # print(catagoriesMatrix[:,0])

        for i in range(0, 50):
            action = 0
            r = 0
            cur = sv
            stock = 0
            prev_state = 0
            res = pd.DataFrame(np.zeros(
                shape=(self.prices.shape[0], 1)), index=self.prices.index, columns={'value'})
            for day in range(0, len(self.prices)):
                stock_price = self.prices.iloc[day][self.tickName]
                state = self.State(catagoriesMatrix[:, day])
                trueState = state+prev_state*self.NumOfStates()
                if day == 0:
                    action = self.Q.querysetstate(trueState)
                else:
                    r = cur+stock*stock_price-res.iloc[day-1][0]
                    action = self.Q.query(trueState, r)
                prev_state = state
                if action == 1:
                    res.iloc[day][0] = cur + stock * stock_price
                    continue
                if action == 2:
                    adjusted_stock_price = stock_price*(1+self.impact)
                    target = self.stock_limit
                elif action == 0:
                    adjusted_stock_price = stock_price*(1-self.impact)
                    target = -self.stock_limit
                if target != stock:
                    cur += -adjusted_stock_price*(target-stock)-self.commission
                    stock = target
                res.iloc[day][0] = cur+stock*stock_price

    def State(self, cuts):
        return np.inner(self.multiplier[:-1], cuts)

    def NumOfStates(self):
        return self.multiplier[-1]

    def CreateBin(self, signal, createBin=True):
        if createBin == True:
            self.catagories[signal], self.bins[signal] = pd.cut(
                self.prices[signal], self.discretize_size, labels=False, retbins=True)
            self.bins[signal][0] = -np.inf
            self.bins[signal][-1] = np.inf
        else:
            self.catagories[signal] = pd.cut(
                self.prices[signal], self.bins[signal], labels=False)

    def SMASignal(self, fast=10, slow=30):
        fastName = indicators.SMA(self.prices, self.tickName, fast)
        slowName = indicators.SMA(self.prices, self.tickName, slow)
        signalName = self.CrossSignal('SMA', fastName, slowName)
        return signalName

    def BollingerBandsSignal(self, window=20, m=2):
        [name, low, high] = indicators.BollingerBands(
            self.prices, self.tickName, window, m)

        signalName = 'BB-Signal'
        self.prices[signalName] = (
            self.prices[self.tickName]-self.prices[low])/(self.prices[high]-self.prices[low])
        return signalName

    def RSISignal(self, window=14, limit=30):
        name = indicators.RSI(self.prices, self.tickName, window)
        signalName = 'RSI-Signal'
        self.prices = self.prices.rename(
            columns={'RSI-'+str(window): signalName})
        return signalName

    def StochasticOscillatorSignal(self, window=14, K=3, D=3):
        [fastName, slowName] = indicators.StochasticOscillator(
            self.prices, self.tickName, window, K, D)
        signalName = self.CrossSignal('SO', fastName, slowName)
        return signalName

        # cross and slow goes below fast => buy signal (1)
        # cross and slow goes above fast => sell signal (-1)
        # return name-Signal in prices

    def CrossSignal(self, name, fastName, slowName):
        signalName = name+'-Signal'
        self.prices[signalName] = self.prices[fastName] - self.prices[slowName]
        return signalName

    # this method should use the existing policy and test it against new data

    def testPolicy(
        self,
        symbol="IBM",
        sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009, 12, 31),
        sv=10000,
    ):
        """  		  	   		  		 			  		 			 	 	 		 		 	
        Tests your learner using data outside of the training data  		  	   		  		 			  		 			 	 	 		 		 	

        :param symbol: The stock symbol that you trained on on  		  	   		  		 			  		 			 	 	 		 		 	
        :type symbol: str  		  	   		  		 			  		 			 	 	 		 		 	
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		  		 			  		 			 	 	 		 		 	
        :type sd: datetime  		  	   		  		 			  		 			 	 	 		 		 	
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		  		 			  		 			 	 	 		 		 	
        :type ed: datetime  		  	   		  		 			  		 			 	 	 		 		 	
        :param sv: The starting value of the portfolio  		  	   		  		 			  		 			 	 	 		 		 	
        :type sv: int  		  	   		  		 			  		 			 	 	 		 		 	
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		  		 			  		 			 	 	 		 		 	
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		  		 			  		 			 	 	 		 		 	
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		  		 			  		 			 	 	 		 		 	
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		  		 			  		 			 	 	 		 		 	
        :rtype: pandas.DataFrame  		  	   		  		 			  		 			 	 	 		 		 	
        """

        # here we build a fake set of trades
        # your code should return the same sort of data
        self.start = sd
        self.end = ed
        self.tickName = symbol
        dates = pd.date_range(self.start - 80*pd.Timedelta(days=1), self.end)
        self.prices = ut.get_data([self.tickName], dates, colname="Adj Close").drop(
            columns=["SPY"])

        self.SMASignal(self.SMA_fast, self.SMA_slow)
        self.BollingerBandsSignal(self.BB_window, self.BB_m)
        self.RSISignal(self.RSI_window, self.RSI_limit)
        self.StochasticOscillatorSignal(self.SO_window, self.SO_K, self.SO_D)

        self.prices = self.prices[self.start:self.end]
        for i in self.signals:
            self.CreateBin(i, createBin=False)

        catagoriesMatrix = np.array([self.catagories[i] for i in self.signals])
        # print(catagoriesMatrix[:,0])

        action = 0
        r = 0
        cur = sv
        stock = 0
        prev_state = 0
        res = pd.DataFrame(np.zeros(
            shape=(self.prices.shape[0], 1)), index=self.prices.index, columns={'value'})
        for day in range(0, len(self.prices)):
            stock_price = self.prices.iloc[day][self.tickName]
            state = self.State(catagoriesMatrix[:, day])
            trueState = state + prev_state * self.NumOfStates()
            action = self.Q.querysetstate(trueState)
            prev_state = state
            if action == 1:
                res.iloc[day][0] = cur + stock * stock_price
                continue
            if action == 2:
                adjusted_stock_price = stock_price*(1+self.impact)
                target = self.stock_limit
            elif action == 0:
                adjusted_stock_price = stock_price*(1-self.impact)
                target = -self.stock_limit
            if target != stock:
                cur += -adjusted_stock_price*(target-stock)-self.commission
                stock = target
            res.iloc[day][0] = cur+stock*stock_price
        return res


if __name__ == "__main__":
    print("One does not simply think up a strategy")

sl = StrategyLearner()
sl.setSMA(10, 34)
sl.setRSI(13, 19)
sl.setSO(20, 6, 1)
sl.setBB(16, 1.47924663)
sl.add_evidence('JPM', dt.datetime(2008, 1, 1),
                dt.datetime(2009, 12, 31), 100000)
res = sl.testPolicy('JPM', dt.datetime(2010, 1, 1),
                dt.datetime(2011, 12, 31), 100000)
print(res.tail(1).values[0][0])
prices = sl.prices
fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 6))
prices[sl.tickName].plot(ax=ax1)
# prices['SO'].plot(ax=ax1)
# prices['K%'].plot(ax=ax1)
# prices['D%'].plot(ax=ax1)
# signresal.plot(ax=ax1)
res.plot(ax=ax2)
ax1.set_xticks([])
ax2.set_xlim([sl.start, sl.end])
ax2.set_xlim([sl.start, sl.end])
plt.show()
