import datetime as dt
import os
import numpy as np
import pandas as pd
from util import get_data, plot_data


def testPolicy(symbol, sd, ed, sv):

    dates = pd.date_range(sd, ed)
    prices = get_data([symbol], dates).drop(
        columns=["SPY"])  # automatically adds SPY

    res = pd.DataFrame(np.zeros(
        shape=(prices.shape[0], 1)), index=prices.index, columns={'trade'})
    cur = 0
    for i in range(0, len(res)-1):
        if prices.iloc[i+1, 0] > prices.iloc[i, 0]:
            res.iloc[i].trade = 1000-cur
            cur = 1000
        else:
            res.iloc[i].trade = -1000-cur
            cur = -1000
    return res


def author():
    """  		  	   		  		 			  		 			 	 	 		 		 	
    :return: The GT username of the student  		  	   		  		 			  		 			 	 	 		 		 	
    :rtype: str  		  	   		  		 			  		 			 	 	 		 		 	
    """
    return "dmiao3"  # replace tb34 with your Georgia Tech username
