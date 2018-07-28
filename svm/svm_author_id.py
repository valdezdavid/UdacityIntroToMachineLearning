#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess

# import
from sklearn import svm
from sklearn.metrics import accuracy_score


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

# Throws out 99% of data for faster training, but loses accuracy (went from 98.4 to 88.4)
# Commenting this out to test the C optimization parameter
# features_train = features_train[:len(features_train)/100] 
# labels_train = labels_train[:len(labels_train)/100] 

# Changed the kernel from linear to rbf
# Changed the C parameter to 1.0 and prediction stays the same
# Changed the C paramater to 10.0 and prediction stays the same
# Changed the C paramater to 100.0 and prediction stays the same
# Changed the C paramater to 1000.0 and prediction went up to 82.3
# Changed the C paramater to 10000.0 and prediction went up to 89.3
clf = svm.SVC(kernel="rbf", C =10000.0)
clf.fit(features_train, labels_train)


pred = clf.predict(features_test)

# Obtaining how many tests predicted a 1 (Chris)
# 877 tests were predicted that Chris wrote them
# this was using the full data, kernel set to rbf, and C = 10000.0
i = 0
counter = 0
for i in range(len(pred)):
	if pred[i] == 1:
		counter += 1

print counter

print accuracy_score(pred, labels_test)

# Printing the prediction for the email at spot x of the list 
# 1 for Chris, Sara is 0
answer = pred[50]

print answer

# After training on all the data, and having the kernel set to rbf the prediction was
# over 99%

#########################################################
### your code goes here ###

#########################################################


