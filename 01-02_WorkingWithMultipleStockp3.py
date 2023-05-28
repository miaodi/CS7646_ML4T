"""
(c) 2015 by Devpriya Dave and Tucker Balch.
"""

"""=================================================================================="""

"""Build a DataFrame in Pandas"""

import pandas as pd

def test_run():
	# Define data range
	start_date = '2010-01-22'
	end_date = '2010-01-26';
	dates=pd.date_range(start_date,end_date)
	
	print (dates)
	print (dates[0])  # get first element of list
	
	# Create an empty dataframe
	df1=pd.DataFrame(index=dates)  # define empty dataframe with these dates as index

	print (df1)
	
	# Read SPY data into temporary dataframe
	# dfSPY = pd.read_csv("data/SPY.csv") # will result in no data because this has index of integers
	# dfSPY = pd.read_csv("data/SPY.csv", index_col="Date", parse_dates=True)
	dfSPY = pd.read_csv("data/SPY.csv", index_col="Date",
						parse_dates=True, usecols=['Date','Adj Close'],
						na_values=['nan'])
	print (dfSPY)
	
	# Join the two dataframes using DataFram.join()
	df1=df1.join(dfSPY)
	print (df1)
	
	# Drop NaN Values
	df1 = df1.dropna()
	print (df1)

if __name__ == "__main__":
    test_run()
