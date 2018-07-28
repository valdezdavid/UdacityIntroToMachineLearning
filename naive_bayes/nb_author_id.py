#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
#import
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()



#########################################################
### your code goes here ###

# Create Gaussian Naive Bayes Classifier
clf = GaussianNB()
# Fit the training data
clf.fit(features_train, labels_train)

# Make a prediction
# Takes in the test features, it will predict labels 
pred = clf.predict(features_test)

# Print the accuracy to see how our classifier did
print accuracy_score(pred, labels_test)

# The accuracy of the prediction is 97.32%

#########################################################


