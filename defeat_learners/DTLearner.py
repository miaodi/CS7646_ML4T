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
class DTLearner(object):
    """  		  	   		  		 			  		 			 	 	 		 		 	
    This is a Linear Regression Learner. It is implemented correctly.  		  	   		  		 			  		 			 	 	 		 		 	

    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		  		 			  		 			 	 	 		 		 	
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.  		  	   		  		 			  		 			 	 	 		 		 	
    :type verbose: bool  		  	   		  		 			  		 			 	 	 		 		 	
    """

    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.model_coefs = np.zeros((2, 2))
        self.maxDepth = 0

    def author(self):
        """  		  	   		  		 			  		 			 	 	 		 		 	
        :return: The GT username of the student  		  	   		  		 			  		 			 	 	 		 		 	
        :rtype: str  		  	   		  		 			  		 			 	 	 		 		 	
        """
        return "dmiao3"  # replace tb34 with your Georgia Tech username

    # pick the best parameter for split
    def PickParam(self, data):
        coeff = abs(np.corrcoef(x=data, rowvar=False)[-1, :-1])
        coeff[np.isnan(coeff)] = 0
        return int(np.argmax(coeff))

    def CompSplitVal(self, data, param):
        return np.median(data[:, param])

    # build tree
    def build_tree(self, data, leaf_size):
        if data.shape[0] <= leaf_size or (data[:, -1] == data[0, -1]).all():
            return (np.asarray([-1, np.mean(data[:, -1]), np.nan, np.nan]), 1)
        else:
            param = self.PickParam(data)
            splitVal = self.CompSplitVal(data, param)
            left_data = data[data[:, param] <= splitVal]
            right_data = data[data[:, param] > splitVal]
            if left_data.shape[0] == 0 or right_data.shape[0] == 0:
                return (np.asarray([-1, np.mean(data[:, -1]), np.nan, np.nan]), 1)
            leftTree, leftDepth = self.build_tree(left_data, leaf_size)
            rightTree, rightDepth = self.build_tree(right_data, leaf_size)
            root = np.asarray(
                [int(param), splitVal, 1, 2 if leftTree.ndim == 1 else leftTree.shape[0]+1])
            root = np.vstack((root, leftTree))
            root = np.vstack((root, rightTree))
            return (root, max(leftDepth, rightDepth)+1)

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
        self.model_coefs, self.maxDepth = self.build_tree(
            data, leaf_size=self.leaf_size)
        # self.model_coefs = self.model_coefs[:,0].astype(int)
        if self.verbose:
            print(self.model_coefs)

    def query(self, points):
        """  		  	   		  		 			  		 			 	 	 		 		 	
        Estimate a set of test points given the model we built.  		  	   		  		 			  		 			 	 	 		 		 	

        :param points: A numpy array with each row corresponding to a specific query.  		  	   		  		 			  		 			 	 	 		 		 	
        :type points: numpy.ndarray  		  	   		  		 			  		 			 	 	 		 		 	
        :return: The predicted result of the input data according to the trained model  		  	   		  		 			  		 			 	 	 		 		 	
        :rtype: numpy.ndarray  		  	   		  		 			  		 			 	 	 		 		 	
        """
        size = points.shape[0]
        res = np.zeros((size))
        for i in range(size):
            pos = 0
            while True:
                if int(self.model_coefs[int(pos), 0]) == -1:
                    res[i] = self.model_coefs[pos, 1]
                    break
                else:
                    if points[i, int(self.model_coefs[pos, 0])] <= self.model_coefs[pos, 1]:
                        pos = pos + int(self.model_coefs[pos, 2])
                    else:
                        pos = pos + int(self.model_coefs[pos, 3])
        return res

    def tableRows(self):
        return self.model_coefs.shape[0]

    def Depth(self):
        return self.maxDepth

if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")
