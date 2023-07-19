import LinRegLearner as lrl  
import BagLearner as bt		

class InsaneLearner(object):
    def __init__(self, verbose = False):
        self.verbose = verbose
        self.learner = bt.BagLearner(learner=lrl.LinRegLearner, kwargs={'verbose': self.verbose}, bags=20)
  		  	   		  		 			  		 			 	 	 		 		 	
    def author(self):  		  	   		 	 			  		 			 	 	 		 		 	
        return "dmiao3"  # replace tb34 with your Georgia Tech username  		  	   

    def add_evidence(self, data_x, data_y):  		  			  		 			 	 	 		 		 	 			 	 	 		 		 	
        self.learner.add_evidence(data_x, data_y)	   	  		 			  		 			 	 	 		 		 			 	    
    			  		 			 	 	 		 		 	
    def query(self, points):  
        return self.learner.query(points)	  