#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 18:07:32 2018

@author: lucien
"""
#Code taken from:
#http://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html

import numpy as np

from time import time
from scipy.stats import randint as sp_randint

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


#%% get some data

#training data
data_train=np.loadtxt('training_data.txt', skiprows=1, delimiter=' ')
y_all=data_train[:,0]
x_all=np.delete(data_train,0,1)

x_train0, x_test0, y_train0, y_test0 = train_test_split(x_all, y_all,   
                                                        test_size=0.05,
                                                        random_state=0)


#%%
# build a classifier
clf = GradientBoostingClassifier()


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


#%% specify parameters and distributions to sample from
param_dist = {"loss": ['deviance','exponential'],
              "learning_rate": [1e-3,1e-2,1e-1,5e-1],
              "n_estimators": [100,200,300,400,500,700,1000],
              "max_depth": sp_randint(1, 15),
              "criterion": ['friedman_mse'],
              "min_samples_split": sp_randint(2, 15),
              "min_samples_leaf":sp_randint(1, 15),
              "max_features":['auto']}

# run randomized search
n_iter_search = 100
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,cv=5,
                                   n_iter=n_iter_search,scoring='accuracy',verbose=3)

start = time()
random_search.fit(x_train0, y_train0)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.cv_results_)


print()
print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
y_true, y_pred = y_test0, random_search.predict(x_test0)
print(classification_report(y_true, y_pred))
print() 

#%% use a full grid over all parameters
param_grid = {"max_depth": [3, None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# run grid search
grid_search = GridSearchCV(clf, param_grid=param_grid)
start = time()
grid_search.fit(X, y)

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)






#%%convert features to tfidf
def tf_idf(X):
    n_rows,n_cols=X.shape
    TFIDF=np.zeros([n_rows,n_cols])
    
    for j in range(0,n_cols):
        num_docs=np.count_nonzero(X[:,j])
        for i in range(0,n_rows):
            TFIDF[i,j]=X[i,j]*np.log(n_rows/num_docs)
    
    return TFIDF        
            
            
            
        

