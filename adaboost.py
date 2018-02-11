#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###############################################################################
# CS 155, Miniproject 1 "Kaggle Competition"
# Adaboost with Randomized Hyperparameter Search
# Lucien Werner & Spencer Gordon
# Feb 8, 2018
###############################################################################

#%% Import modules

import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import Normalizer    
import time  

#%% Import training data

data_train=np.loadtxt('training_data.txt', skiprows=1, delimiter=' ')
y_all=data_train[:,0] #get last 1000 columns
x_all=np.delete(data_train,0,1) #get first column

#%% Pre-processing on X training data

#note that these are four methods we tried to normalize the data
#only a single methods was used at a time

# (1) normalize data (TFIDF)
tfdif=TfidfTransformer(norm='l1')
x_tfdif=tfdif.fit_transform(x_all)

# (2) log normalization
x_log=np.log(1+x_all)

# (3) binary normalization (convert all non-zero entries to 1)
binar=Binarizer()
x_bin=binar.fit_transform(x_all)

# (4) normalize w.r.t each feature
normal=Normalizer(norm='l2')
x_normal=normal.fit_transform(x_all)

#generate a test-train split for validation (does not mean cross-validation)
# note that the random state is set for exact recall
x_train0, x_test0, y_train0, y_test0 = train_test_split(x_all, y_all,   
                                                        test_size=0.05,
                                                        random_state=0) 

#%% Utility function to report best scores

#Code adapted from:
#http://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html

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

#%% Build and train AdaBoost classifier

#declare a classifier
clf = AdaBoostClassifier()

#specify distribution of hyperparameters
param_dist = {'learning_rate':[1,1e-1,1e-2,1e-3],
              'n_estimators':[10,50,100,150,200]}


# run randomized grid search over parameter distribution (full grid search took too long)

n_iter_search = 100
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,cv=5,
                                   n_iter=n_iter_search,scoring='accuracy',verbose=3)

#report accuracy for each split/set of parameters
start = time()
random_search.fit(x_train0, y_train0)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.cv_results_)


#validate accuracy on test split
print()
print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
y_true, y_pred = y_test0, random_search.predict(x_test0)
print(classification_report(y_true, y_pred))
print() 

#%% Preliminary validation of this classifier told us it was not good enough to proceed. 

#No predictions made with this model
