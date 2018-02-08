#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 16:13:04 2018

@author: lucien
"""


import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC 
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import Normalizer
#%%convert features to tfidf
def tf_idf(X):
    n_rows,n_cols=X.shape
    TFIDF=np.zeros([n_rows,n_cols])
    
    for j in range(0,n_cols):
        num_docs=np.count_nonzero(X[:,j])
        for i in range(0,n_rows):
            TFIDF[i,j]=X[i,j]*np.log(n_rows/num_docs)
    
    return TFIDF        
            
#%% Import data

#training data
data_train=np.loadtxt('training_data.txt', skiprows=1, delimiter=' ')
y_all=data_train[:,0]
x_all=np.delete(data_train,0,1)

#%% pre-processing


#normalize data (TFIDF)
#tfdif=TfidfTransformer(norm='l1')
#x_all_norm=tfdif.fit_transform(x_all)

#x_all_log=np.log(1+x_all) #log preprocessing

#binar=Binarizer()
#x_all_bin=binar.fit_transform(x_all)

#normal=Normalizer(norm='l2')
#x_all_norm=normal.fit_transform(x_all)



x_train0, x_test0, y_train0, y_test0 = train_test_split(x_all, y_all,   
                                                        test_size=0.05,
                                                        random_state=0)
#%%

#SVC()
#tuned_parameters = [{'kernel': ['poly'], 'gamma': [1e-3],'degree':[2],
                     'C': [1],'verbose':[1]}]
                    #{'kernel': ['linear'], 'C': [1, 10, 100]}]

LogisticRegression()
tuned_parameters = [{'solver':['saga'],'C':[1,5e-1,1e-1,1e-2],'max_iter':[300]}]

clf = GridSearchCV(LogisticRegression(), tuned_parameters, cv=5,
                       scoring='accuracy', verbose=3)
clf.fit(x_train0, y_train0)

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
print(classification_report(y_true, y_pred,digits=5))
print() 


#%% run prediction on official test data


#first train on all data
bestclf=LogisticRegressionCV(Cs=[0.1],cv=5,solver='saga',scoring='accuracy',max_iter=300)
bestclf.fit(x_all, y_all)
#y_true, y_pred = y_test0, bestclf.predict(x_test0)
#print(classification_report(y_true, y_pred,digits=5))



#%%




X_TEST=np.loadtxt('test_data.txt', skiprows=1, delimiter=' ')
X_TEST_LOG=np.log(1+X_TEST)

Y_PRED=bestclf.predict(X_TEST)

output_array=np.column_stack((np.linspace(1,len(Y_PRED),len(Y_PRED)),Y_PRED))

#write data to file
import time
t=time.localtime()
timestamp=time.strftime('%Y%m%d_%H%M%S',t)
filename='submission_'+ timestamp +'.txt'
np.savetxt(filename,output_array.astype(int),fmt='%i',delimiter=',',header='Id,Prediction')


