""""""
"""  		  	   		  		 			  		 			 	 	 		 		 	
A simple wrapper for linear regression.  (c) 2015 Tucker Balch  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
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




import numpy as np
import random
def sampleGen(data):
    size = data.shape[0]
    res = np.empty((0, data.shape[1]), float)
    for i in range(size):
        res = np.vstack((res, data[random.randint(0, size-1), :]))
    return res


class BagLearner(object):
    def __init__(self, learner, kwargs, bags, boost = False, verbose = False):
        self.bags = bags
        self.boost = boost
        self.verbose = verbose
        self.learners = []
        for i in range(bags):
            self.learners.append(learner(**kwargs))

    def author(self):
        """  		  	   		  		 			  		 			 	 	 		 		 	
        :return: The GT username of the student  		  	   		  		 			  		 			 	 	 		 		 	
        :rtype: str  		  	   		  		 			  		 			 	 	 		 		 	
        """
        return "dmiao3"  # replace tb34 with your Georgia Tech username

    def add_evidence(self, data_x, data_y):
        """  		  	   		  		 			  		 			 	 	 		 		 	
        Add training data to learner  		  	   		  		 			  		 			 	 	 		 		 	

        :param data_x: A set of feature values used to train the learner  		  	   		  		 			  		 			 	 	 		 		 	
        :type data_x: numpy.ndarray  		  	   		  		 			  		 			 	 	 		 		 	
        :param data_y: The value we are attempting to predict given the X data  		  	   		  		 			  		 			 	 	 		 		 	
        :type data_y: numpy.ndarray  		  	   		  		 			  		 			 	 	 		 		 	
        """

        data_y = data_y[:, np.newaxis]
        data = np.append(data_x, data_y, axis=1)

        for learner in self.learners:
            newData = sampleGen(data)
            learner.add_evidence(newData[:, :-1], newData[:, -1])

    def query(self, points):
        size = points.shape[0]
        res = np.zeros((size))
        for learner in self.learners:
            res += learner.query(points)
        return res/self.bags
