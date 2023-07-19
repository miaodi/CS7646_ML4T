""""""
"""  		  	   		  		 			  		 			 	 	 		 		 	
Test a learner.  (c) 2015 Tucker Balch  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
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
"""
from numpy import linalg as LA
import matplotlib.pyplot as plt
import math
import sys
import numpy as np
import LinRegLearner as lrl
import DTLearner as dt
import BagLearner as bt
import RTLearner as rt
import time

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python testlearner.py <filename>")
        sys.exit(1)
    data = np.genfromtxt(sys.argv[1], delimiter=',')
    if np.isnan(data[0, 0]):
        data = np.delete(data, 0, 0)
        data = np.delete(data, 0, 1)

    # compute how much of the data is training and testing
    train_rows = int(0.6 * data.shape[0])
    test_rows = data.shape[0] - train_rows

    # np.random.seed(123456789)
    # permute data
    data = np.random.permutation(data)
    # separate out training and testing data
    train_x = data[:train_rows, 0:-1]
    train_y = data[:train_rows, -1]
    test_x = data[train_rows:, 0:-1]
    test_y = data[train_rows:, -1]

    print(f"{test_x.shape}")
    print(f"{test_y.shape}")

    # # create a learner and train it
    # learner = lrl.LinRegLearner(verbose=True)  # create a LinRegLearner
    # learner.add_evidence(train_x, train_y)  # train it
    # print(learner.author())

    # # evaluate in sample
    # pred_y = learner.query(train_x)  # get the predictions
    # rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
    # print()
    # print("In sample results")
    # print(f"RMSE: {rmse}")
    # c = np.corrcoef(pred_y, y=train_y)
    # print(f"corr: {c[0,1]}")

    # # evaluate out of sample
    # pred_y = learner.query(test_x)  # get the predictions
    # rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
    # print()
    # print("Out of sample results")
    # print(f"RMSE: {rmse}")
    # c = np.corrcoef(pred_y, y=test_y)
    # print(f"corr: {c[0,1]}")
    # Experiment 1
    def experiment1():
        max_leaf = 50
        train_err = []
        test_err = []
        depth = []
        for i in range(1, max_leaf):
            learner = dt.DTLearner(leaf_size=i, verbose=False)
            learner.add_evidence(train_x, train_y)
            train_pred = learner.query(train_x)
            test_pred = learner.query(test_x)
            train_err.append(LA.norm(train_y-train_pred, 2) /
                             math.sqrt(train_rows))
            test_err.append(LA.norm(test_y-test_pred, 2)/math.sqrt(test_rows))
            depth.append(learner.Depth())

        plt.figure(0)
        figure, ax = plt.subplots()
        plt.plot(train_err, color="blue",
                 linestyle='dashed', label='in sample')
        plt.plot(test_err, color="blue", label='out of sample')

        ax2 = ax.twinx()
        # ax2.plot(time, temp, '-r', label = 'temp')
        ax2.plot(depth, color="green", label='depth')

        ax.legend(loc=0)
        ax2.legend(loc=0)
        ax.set_xlabel('Leaf size')
        ax.set_ylabel('RMSE')
        ax2.set_ylabel('Depth')
        ax2.set_ylim(2, 12)
        ax.set_title(
            'Relation among leaf size, tree depth and overfitting of DTLearner')
        fig = plt.gcf()
        fig.savefig('experiment1', dpi=300)

    # Experiment 2
    def experiment2():
        max_leaf = 50
        train_err = []
        test_err = []
        plt.figure(1)
        for i in [1, 2, 4, 8, 16]:
            for j in range(1, max_leaf):
                learner = bt.BagLearner(learner=dt.DTLearner, kwargs={
                                        'leaf_size': j, 'verbose': False}, bags=i)
                learner.add_evidence(train_x, train_y)
                train_pred = learner.query(train_x)
                test_pred = learner.query(test_x)
                # train_err.append(LA.norm(train_y-train_pred, 2) /
                #                  math.sqrt(train_rows))
                test_err.append(LA.norm(test_y-test_pred, 2) /
                                math.sqrt(test_rows))
            plt.plot(test_err)
            test_err = []

        plt.legend(['1 bags', '2 bags', '4 bags', '8 bags', '16 bags'])
        plt.gca().set_xlabel('Leaf size')
        plt.gca().set_ylabel('RMSE')
        plt.gca().set_title('Relation between leaf size and overfitting of BagLearner+DTLearner')
        fig = plt.gcf()
        fig.savefig('experiment2', dpi=300)

    # Experiment 3
    def experiment3():
        max_leaf = 50
        train_err_dt = []
        test_err_dt = []
        rows_dt = []
        rows_rt = []
        time_dt = []
        time_rt = []
        for i in range(1, max_leaf):
            learner = dt.DTLearner(leaf_size=i, verbose=False)
            t = time.process_time()
            learner.add_evidence(train_x, train_y)
            time_dt.append(time.process_time()-t)
            train_pred = learner.query(train_x)
            test_pred = learner.query(test_x)
            train_err_dt.append(LA.norm(train_y-train_pred, np.inf))
            test_err_dt.append(LA.norm(test_y-test_pred, np.inf))
            rows_dt.append(learner.Depth())
        plt.figure(2)
        plt.plot(train_err_dt, color="blue", linestyle='dashed')
        plt.plot(test_err_dt, color="blue")

        train_err_rt = []
        test_err_rt = []
        for i in range(1, max_leaf):
            learner = rt.RTLearner(leaf_size=i, verbose=False)
            t = time.process_time()
            learner.add_evidence(train_x, train_y)
            time_rt.append(time.process_time()-t)
            train_pred = learner.query(train_x)
            test_pred = learner.query(test_x)
            train_err_rt.append(LA.norm(train_y-train_pred, np.inf))
            test_err_rt.append(LA.norm(test_y-test_pred, np.inf))
            rows_rt.append(learner.Depth())
        plt.plot(train_err_rt, color="red", linestyle='dashed')
        plt.plot(test_err_rt, color="red")

        plt.legend(['in sample DT', 'out of sample DT',
                   'in sample RT', 'out of sample RT'])
        plt.gca().set_xlabel('Leaf size')
        plt.gca().set_ylabel('$\|\|y-y_{train}\|\|_{\inf}$')
        plt.gca().set_title('Error comparison between DTLearner and RTLearner.')
        fig = plt.gcf()
        fig.savefig('experiment3', dpi=300)

        plt.figure(3)
        plt.plot(rows_dt, color="blue", label= "depth of DTLearner")
        plt.plot(rows_rt, color="red",  label= "depth of RTLearner")
        plt.gca().set_xlabel('Leaf size')
        plt.gca().set_ylabel('Depth')
        plt.gca().set_title('Relation between leaf size and Tree depth')
        plt.legend(loc=0)
        plt.savefig('experiment3_2', dpi=300)

        plt.figure(4)
        plt.plot(time_dt, color="blue", label= "time to train DTLearner")
        plt.plot(time_rt, color="red",  label= "time to train RTLearner")
        plt.gca().set_xlabel('Leaf size')
        plt.gca().set_ylabel('Time to train learner model (s)')
        plt.gca().set_title('Time consumption in training model.')
        plt.legend(loc=0)
        plt.savefig('experiment3_3', dpi=300)

    experiment1()
    experiment2()
    experiment3()
