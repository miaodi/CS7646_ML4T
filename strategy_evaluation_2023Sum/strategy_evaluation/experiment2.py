

import datetime as dt
import random
import pandas as pd
import util as ut
import indicators
import numpy as np
import QLearner
import matplotlib.pyplot as plt
import ManualStrategy
import os
import StrategyLearner


def author():
    return ('dmiao3')


def exec():
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    sl = StrategyLearner.StrategyLearner(impact=0.000)
    sl.add_evidence('JPM', sd, ed, 100000)
    trades1 = sl.testPolicy('JPM', sd, ed, 100000)
    optResSL1 = sl.portValue()

    sl = StrategyLearner.StrategyLearner(impact=0.005)
    sl.add_evidence('JPM', sd, ed, 100000)
    trades2 = sl.testPolicy('JPM', sd, ed, 100000)
    optResSL2 = sl.portValue()

    sl = StrategyLearner.StrategyLearner(impact=0.01)
    sl.add_evidence('JPM', sd, ed, 100000)
    trades3 = sl.testPolicy('JPM', sd, ed, 100000)
    optResSL3 = sl.portValue()

    fig, (ax1) = plt.subplots(figsize=(10, 6))
    ymin, ymax = ax1.get_ylim()
    optResSL1.plot(ax=ax1, color='purple')
    optResSL2.plot(ax=ax1, color='red')
    optResSL3.plot(ax=ax1, color='green')
    ax1.legend(['impact=0.000', 'impact=0.005',
               'impact=0.010'], loc='upper left')
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Normalized value")
    plt.savefig(os.path.dirname(__file__) +
                "/plots/"+'experiment2_return.eps', format='eps')

    print("Num of trades for impact= 0: {}".format(
        trades1.astype(bool).sum(axis=0)))
    print("Num of trades for impact= 0.005: {}".format(
        trades2.astype(bool).sum(axis=0)))
    print("Num of trades for impact= 0.010: {}".format(
        trades3.astype(bool).sum(axis=0)))

    print("Cumulative return for impact= 0: {}".format(
        optResSL1.iloc[-1]['value']-1))
    print("Cumulative return for impact= 0.005: {}".format(
        optResSL2.iloc[-1]['value']-1))
    print("Cumulative return for impact= 0.010: {}".format(
        optResSL3.iloc[-1]['value']-1))


if __name__ == "__main__":
    exec()