#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###############################################################################
# CS 155, Miniproject 1 "Kaggle Competition"
# Naive Bayesian Classifier
# Lucien Werner & Spencer Gordon
# Feb 8, 2018
###############################################################################

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
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

#%% Train classifier with n_splits-fold cross validation

clf = GaussianNB()
n_splits=3 #number of cross-validation splits
kf = KFold(n_splits=n_splits,shuffle=True,random_state=0) #get k splits
kf.get_n_splits(x_train0) #verify splits

sum=0
#train model over all splits and validate accuracy
for train_index, test_index in kf.split(x_train0):
    
    x_train, x_test = x_train0[train_index], x_train0[test_index]
    y_train, y_test = y_train0[train_index], y_train0[test_index]

    clf.fit(x_train0,y_train0)
    y_pred= clf.predict(x_test0)
    sum += accuracy_score(y_test0,y_pred, normalize = True)

accuracy=sum/n_splits   
print()
print(accuracy)


#%% Make predictions on test data for submission

#load test_data.txt
X_TEST=np.loadtxt('test_data.txt', skiprows=1, delimiter=' ')

#normalize data according to the method chosen for model training (log was best)
X_TEST_LOG=np.log(1+X_TEST)

#predict the y values
Y_PRED=clf.predict(X_TEST)

#make array with row indices as column 1 and predictions as column 2
output_array=np.column_stack((np.linspace(1,len(Y_PRED),len(Y_PRED)),Y_PRED))

#write data to file with timestamp
t=time.localtime()
timestamp=time.strftime('%Y%m%d_%H%M%S',t)
filename='submission_'+ timestamp +'.txt'
np.savetxt(filename,output_array.astype(int),fmt='%i',delimiter=',',header='Id,Prediction')

# !!still need to remove sharp # from header line in file manually before submission !!



