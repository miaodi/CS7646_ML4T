import datetime as dt
import os
import numpy as np
import pandas as pd
from util import get_data, plot_data
import marketsimcode
import TheoreticallyOptimalStrategy
import indicators
import matplotlib.pyplot as plt


def author():
    """  		  	   		  		 			  		 			 	 	 		 		 	
    :return: The GT username of the student  		  	   		  		 			  		 			 	 	 		 		 	
    :rtype: str  		  	   		  		 			  		 			 	 	 		 		 	
    """
    return "dmiao3"  # replace tb34 with your Georgia Tech username

if __name__ == "__main__":
    # # part 1
    df_trades = TheoreticallyOptimalStrategy.testPolicy(
        'JPM', dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31), 100000)

    orders = pd.DataFrame(index=df_trades.index,
                        columns=["Symbol", "Order", "Shares"])
    # change df_trades to fit compute_portvals's api
    orders.index.name = 'Date'
    for i in range(0, len(df_trades)):
        orders.iloc[i].Symbol = "JPM"
        if df_trades.iloc[i, 0] > 0:
            orders.iloc[i].Order = "BUY"
            orders.iloc[i].Shares = abs(df_trades.iloc[i, 0])
        else:
            orders.iloc[i].Order = "SELL"
            orders.iloc[i].Shares = abs(df_trades.iloc[i, 0])

    portvals = marketsimcode.compute_portvals(orders, 100000, 0, 0)
    portvals = portvals/portvals.iloc[0]
    portvals['value'].plot.line(color='red')

    # benchmark
    benchmark = 1000*get_data(["JPM"], portvals.index,
                            addSPY=True, colname="Adj Close").drop(columns="SPY")
    rest = 100000-benchmark.iloc[0]['JPM']
    benchmark['JPM'] = benchmark['JPM']+rest
    benchmark["JPM"] = benchmark / benchmark.iloc[0]
    benchmark['JPM'].plot.line(color='purple')
    plt.gca().set_xlabel('Date')
    plt.gca().set_ylabel("Normalized Price")
    plt.title("TOS vs benchmark")
    plt.legend(['TOS', 'benchmark'])
    plt.savefig("./tos.eps", format='eps')

    # ------------- Tables ------------- #
    # Stats
    tos_daily_return = portvals.pct_change().dropna()
    bench_daily_return = benchmark.pct_change().dropna()
    # bench
    bench_std = round(bench_daily_return['JPM'].std(), 6)
    bench_cum_rets = round(benchmark.iloc[-1]['JPM']-1, 6)
    bench_avg_rets = round(bench_daily_return['JPM'].mean(), 6)
    # tos
    tos_std = round(tos_daily_return['value'].std(), 6)
    tos_cum_rets = round(portvals.iloc[-1]['value']-1, 6)
    tos_avg_rets = round(tos_daily_return['value'].mean(), 6)

    headers = ['Portfolio', 'STD', 'Cummulative Rets', 'Average Rets']
    rows = [['benchmark', bench_std, bench_cum_rets, bench_avg_rets],
            ['TOS portfolio', tos_std, tos_cum_rets, tos_avg_rets]]

    df = pd.DataFrame(data=rows, columns=headers)
    df.to_csv(r'metric.txt', header=True, index=None, sep='\t')


    indicators.plots('JPM',  dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31))