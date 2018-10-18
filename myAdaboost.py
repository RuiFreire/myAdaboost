##############################################################
################### My Adaboost Class - Definition
##############

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_array
from sklearn.utils import check_X_y
from sklearn.preprocessing import MinMaxScaler
import copy
import numpy as np
import pandas as pd
import sys

class myAdaboost(BaseEstimator, ClassifierMixin):  


    
    def __init__(self, threshold=0.5, number_of_iterations=50, classifier=None):
        
        self.threshold = threshold
        self.T = number_of_iterations
        self.weak_learners_list = []
        self.classifier = classifier

    def check_y(self, y):
        
        if type(y) == type(pd.Series([])):
            
            if pd.isnull(y).sum() > 0:
                raise NameError("There are None ou Nan values in y")
    
            if not ( np.array_equal( y.unique(), np.array([1, 0]) )  or  np.array_equal( y.unique(), np.array([0, 1]) ) ):
                raise NameError( "y must be a pandas Series or an 1d numpy array with numeric binary values: (0,1)")
            
            
        if type(y) == type(np.array([])):       

            if pd.isnull(pd.Series( y ) ).sum() > 0:
                raise NameError("There are None ou Nan values in y")
            
            if y.dtype == "object":
                raise NameError("y is an array of type objcet. y must be a pandas Series or an 1d numpy array with numeric binary values: (0,1) ")
            
            if not ( np.array_equal(np.unique(y), np.array([0, 1])) or  np.array_equal(np.unique(y), np.array([1, 0])) ):
                raise NameError("y must be a pandas Series or an 1d numpy array with numeric binary values: (0,1)")       
    
    
    def fit(self, X=None, y=None):
        
        
        # calling a method to ensure that "y is a pandas Series or an 1d numpy array with numeric binary values: (0,1)"
        self.check_y(y)
        
        y = y.copy(deep=True)
        
        X, y = check_X_y(X, y)
        y[ y == 0 ] = -1
        
        weights = np.array( [1/len(y)]*len(y) )
        
        np.random.seed(seed=123)
        
        for t in range(self.T):
            
            dic = {}
                  
            train_set_index = np.random.choice(a=len(y), size=len(y), p=weights, replace=True)
                 
            self.classifier.fit( X[train_set_index], y[train_set_index] )
            new_predictions = self.classifier.predict( X )
            
            new_error = sum( weights*( new_predictions != y ) )
        
            new_cl = copy.deepcopy( self.classifier )
            
            alpha = 0.5*np.log( ( 1-new_error )/new_error )
        
            sys.stdout.write('\rIteration:' + str(t) + " | error:" + str(new_error) + " | alpha:" + str(alpha)  )
        
            dic["weak_learner"] = new_cl
            dic["alpha"] = alpha
        
            preNormWeights = weights * np.exp( -1*dic["alpha"]*(y*new_predictions) )
        
            weights = preNormWeights/preNormWeights.sum()
            weights = np.array(weights)
            
            self.weak_learners_list.append(dic)
        
        return self
        
    
    def predict_proba(self, X):
        
        sum_predictions = np.matrix( [0.0]*len(X) )
        
        for weak_learner in self.weak_learners_list:
            predictions = weak_learner["weak_learner"].predict( X )
            sum_predictions += weak_learner["alpha"]*predictions
    
        mMscaler = MinMaxScaler()

        mMscaler.fit(sum_predictions.T)

        sum_predictions_norm = mMscaler.transform(sum_predictions.T)
        
        sum_predictions_norm = np.concatenate( ( (1-sum_predictions_norm), sum_predictions_norm), axis=1 )
        
        return np.array(sum_predictions_norm)
  
    
    def predict(self, X):
        
        predictions = self.predict_proba(X)[:,1]
        
        predictions[ predictions >= self.threshold ] = 1
        predictions[ predictions < self.threshold ] = 0

        return predictions
    
    def score(self, X=None, y=None):
        
        predictions = self.predict(X)
        acc = accuracy_score(y, predictions, normalize=True, sample_weight=None)
        return  acc