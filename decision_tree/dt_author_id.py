#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess

from sklearn.metrics import accuracy_score
from sklearn import tree
### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
# Creates a Decision Tree Classifier with the minimum number of samples split is 40
clf = tree.DecisionTreeClassifier(min_samples_split =40)
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

# We'll get the number of features to try to speed up the algorithm
numFeatures = len(features_train[0])

# When percentile was 10 in selector = SelectPercentile(f_classif, percentile=10)
# there was a total of 3785 features.
# After changing the percentile to 1, the number of features reduced to only 379
# More features = more complex decision tree
# The accuracy was 0.967 when only having 1% of the total available features
# The accuracy was 0.977 when only having 10% of the total available features 

print accuracy_score(pred, labels_test)
print numFeatures


#########################################################


