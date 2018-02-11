#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###############################################################################
# CS 155, Miniproject 1 "Kaggle Competition"
# Logistic Regression and Support Vector Classifier Models
# Lucien Werner & Spencer Gordon
# Feb 8, 2018
###############################################################################

#%% Import modules
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC 
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

#%% Perform Gridsearch over hyperparameters of the models

#multiple classifiers (only one deployed at a time)

#support vector classifier
clf=SVC()
clf_params=[{'kernel':['poly'],'gamma':[1e-3],'degree':[2],'C': [1],'verbose':[1]},
                        {'kernel': ['linear'],'gamma':[1e-3],'C':[1,10,100]},
                        {'kernel': ['rbf'],'gamma':[1e-3],'C':[1,10]}]

#logistic regression classifier
clf=LogisticRegression()
tuned_parameters = [{'solver':['sag','saga'],'C':[1,5e-1,1e-1,1e-2,1e-3,1e-4],'max_iter':[100,200,300]}]

#declare GridSearch object with 5 cross-validation folds, clf, and clf_params
grid_clf = GridSearchCV(clf, clf_params, cv=5,scoring='accuracy', verbose=3)

#fit the model to the training set
grid_clf.fit(x_train0, y_train0) #train on training split

#output training results (code adapted from "http://scikit-learn.org/stable/modules/grid_search.html")
print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
means = grid_clf.cv_results_['mean_test_score']
stds = grid_clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, grid_clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))

#
print()
print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
y_true, y_pred = y_test0, grid_clf.predict(x_test0)
print(classification_report(y_true, y_pred,digits=5))

#%% Train model with best parameters on all the entire training set (NO test-train split)

#parameters are determined from grid_clf.best_params_. Use these in model training
bestclf=LogisticRegressionCV(Cs=[1e-1],cv=5,solver='saga',scoring='accuracy',max_iter=300)
bestclf.fit(x_all, y_all)

# !!cannot validate the model performance because there is no test data with answers!!

#%% Make predictions on test data for submission

#load test_data.txt
X_TEST=np.loadtxt('test_data.txt', skiprows=1, delimiter=' ')

#normalize data according to the method chosen for model training (log was best)
X_TEST_LOG=np.log(1+X_TEST)

#predict the y values
Y_PRED=bestclf.predict(X_TEST)

#make array with row indices as column 1 and predictions as column 2
output_array=np.column_stack((np.linspace(1,len(Y_PRED),len(Y_PRED)),Y_PRED))

#write data to file with timestamp
t=time.localtime()
timestamp=time.strftime('%Y%m%d_%H%M%S',t)
filename='submission_'+ timestamp +'.txt'
np.savetxt(filename,output_array.astype(int),fmt='%i',delimiter=',',header='Id,Prediction')

# !!still need to remove sharp # from header line in file manually before submission !!