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

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import AdaBoostClassifier
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
clf = AdaBoostClassifier()


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
param_dist = {'learning_rate':[1,1e-1,1e-2,1e-3],
              'n_estimators':[10,50,100,150,200]}


# run randomized search
n_iter_search = 10
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
param_grid = {"max_depth": [1,2,3, None],
              "max_features": [20,30,300,320,340],
              "min_samples_split": [2, 3,30],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True],
              "criterion": ["entropy"],
              'n_estimators':[150,250]}

# run grid search
grid_search = GridSearchCV(clf, param_grid=param_grid,scoring='accuracy',cv=5,verbose=3)
grid_search.fit(x_train0,y_train0)

print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))

print()
print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
y_true, y_pred = y_test0, clf.predict(x_test0)
print(classification_report(y_true, y_pred))
print() 





#%%convert features to tfidf
def tf_idf(X):
    n_rows,n_cols=X.shape
    TFIDF=np.zeros([n_rows,n_cols])
    
    for j in range(0,n_cols):
        num_docs=np.count_nonzero(X[:,j])
        for i in range(0,n_rows):
            TFIDF[i,j]=X[i,j]*np.log(n_rows/num_docs)
    
    return TFIDF        
            
            


#%%
    param_grid = {"max_depth": [1,2,3,4,5,6,7, None],
              "max_features": [1, 3, 10,20],
              "min_samples_split": [2, 3, 10,15,30],
              "min_samples_leaf": [1, 3, 10,15,30],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"],
              'n_estimators':[5,10,30,50,70,90,110,130,150]}
            
        

