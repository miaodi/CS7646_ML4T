import datetime as dt
import os
import numpy as np
import pandas as pd
from util import get_data, plot_data
import matplotlib.pyplot as plt

def author():
    """  		  	   		  		 			  		 			 	 	 		 		 	
    :return: The GT username of the student  		  	   		  		 			  		 			 	 	 		 		 	
    :rtype: str  		  	   		  		 			  		 			 	 	 		 		 	
    """
    return "dmiao3"  # replace tb34 with your Georgia Tech username

def SMA(prices, symbol, window=20):
    prices['SMA-'+str(window)] = prices[symbol].rolling(window).mean()


def BollingerBands(prices, symbol, window=20, m=2):
    SMA(prices, symbol, window)
    std = prices[symbol].rolling(window).std()
    prices['BOLU-'+str(window)] = prices['SMA-'+str(window)]+m*std
    prices['BOLD-'+str(window)] = prices['SMA-'+str(window)]-m*std
    prices['SMA-'+str(window)] = prices[symbol].rolling(window).mean()
    prices['BB'] = ((prices[symbol]-prices['SMA-'+str(window)])/(m*std))*100
    prices["-100"] = -100
    prices["100"] = 100


def RSI(prices, symbol, window=14):
    diff = prices[symbol].diff().dropna()
    diffUp = diff.copy()
    diffDown = diff.copy()
    diffUp[diffUp < 0] = 0
    diffDown[diffDown > 0] = 0
    prices['RSI-'+str(window)] = 100-100 / \
        (1+diffUp.rolling(window).mean()/abs(diffDown.rolling(window).mean()))

    prices["Overbought"] = 70
    prices["Oversold"] = 30


def StochasticOscillator(prices, symbol, window=21, K=14, D=14):
    Low = prices[symbol].rolling(window).min()
    High = prices[symbol].rolling(window).max()
    prices['SO'] = (prices[symbol]-Low)/(High-Low)*100
    prices['K%'] = prices['SO'].rolling(K).mean()
    prices['D%'] = prices['K%'].rolling(D).mean()
    prices["80"] = 80
    prices["20"] = 20

def VPT(prices, symbol, window=7, window2=14):
    returns = prices[symbol].pct_change().fillna(0)
    vp = returns*prices['Volume']
    prices['VPT'] = vp.cumsum()
    prices['VPT-SMA-' + str(window)] = prices['VPT'].rolling(window).mean()
    prices['VPT-SMA-' + str(window2)] = prices['VPT'].rolling(window2).mean()

def plots(symbol, sd, ed):
    dates = pd.date_range(sd - 100*pd.Timedelta(days=1), ed)
    # print(dates)
    prices = get_data([symbol], dates, colname="Adj Close").drop(
        columns=["SPY"])  # automatically adds SPY
    volume = get_data([symbol], dates, colname="Volume").drop(
        columns=["SPY"])  # automatically adds SPY
    prices['Volume'] = volume
    # prices = prices.div(prices.iloc[0])
    SMA(prices, symbol, 20)
    SMA(prices, symbol, 50)
    BollingerBands(prices, symbol)
    RSI(prices, symbol)
    StochasticOscillator(prices, symbol, 14, 3, 3)
    VPT(prices, symbol)
    # golden cross
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel('Date')
    ax.set_ylabel("JPM Price ($)")
    prices[symbol].plot()
    prices['SMA-20'].plot()
    prices['SMA-50'].plot()
    ax.set_xlim([sd, ed])
    plt.legend(['Adjusted Close Prices', 'SMA-20', 'SMA-50'])
    plt.title("Simple Moving Average (SMA)")
    plt.savefig("./sma.eps", format='eps')
    # plt.show()
    # fig.show()

    # RSI
    fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 6))
    prices[symbol].plot(ax=ax1)
    prices['RSI-14'].plot(ax=ax2)
    prices['Overbought'].plot(ax=ax2)
    prices['Oversold'].plot(ax=ax2)
    ax1.set_xticks([])
    ax1.set_ylabel("JPM Price ($)")
    ax2.set_ylabel("RSI (%)")
    ax2.set_xlabel("Date")
    ax1.set_xlim([sd, ed])
    ax2.set_xlim([sd, ed])
    fig.suptitle('Relative Strength Index (RSI)')
    plt.legend(['RSI', 'Overbought', 'Oversold'])
    plt.savefig("./rsi.eps", format='eps')

    # Bollinger Band
    fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 6))
    prices[symbol].plot(ax=ax1)
    prices['SMA-20'].plot(ax=ax1)
    prices['BOLU-20'].plot(ax=ax1)
    prices['BOLD-20'].plot(ax=ax1)
    prices['BB'].plot(ax=ax2)
    prices['100'].plot(ax=ax2)
    prices['-100'].plot(ax=ax2)
    ax1.legend(['price', 'SMA-20', '+2std', '-2std'])
    ax2.legend(['BB', 'BOLU', 'BOLD'])
    ax1.set_xticks([])
    ax1.set_ylabel("JPM Price ($)")
    ax2.set_ylabel("Percentage (%)")
    ax2.set_xlabel("Date")
    ax1.set_xlim([sd, ed])
    ax2.set_xlim([sd, ed])
    fig.suptitle('Bollinger Band')
    plt.savefig("./bol.eps", format='eps')

    # Stochastic Oscillator
    fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 6))
    prices[symbol].plot(ax=ax1)
    prices['K%'].plot(ax=ax2)
    prices['D%'].plot(ax=ax2)
    prices['80'].plot(ax=ax2)
    prices['20'].plot(ax=ax2)
    ax1.set_xticks([])
    ax1.set_ylabel("JPM Price ($)")
    ax2.set_ylabel("Percentage (%)")
    ax2.set_xlabel("Date")
    ax1.set_xlim([sd, ed])
    ax2.set_xlim([sd, ed])
    fig.suptitle('Stochastic Oscillator')
    plt.legend(['K%', 'D%', 'Overbought', 'Oversold'],
               loc='upper right', bbox_to_anchor=(1., .75))
    plt.savefig("./so.png", format='png')
    
    #VPT
    fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 6))
    prices[symbol].plot(ax=ax1)
    prices['VPT'].plot(ax=ax2)
    prices['VPT-SMA-7'].plot(ax=ax2)
    # prices['VPT-SMA-14'].plot(ax=ax2)
    ax1.set_xticks([])
    ax1.set_ylabel("JPM Price ($)")
    ax2.set_ylabel("Volume Price")
    ax2.set_xlabel("Date")
    ax1.set_xlim([sd, ed])
    ax2.set_xlim([sd, ed])
    fig.suptitle('Volume Price Trend')
    plt.savefig("./vpt.eps", format='eps')


