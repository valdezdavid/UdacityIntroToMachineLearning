#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
features_train, labels_train, features_test, labels_test = makeTerrainData()

# Having n_neighbors = 3 and algorithm set to 'auto', it was able to reach the
# desired accuracy in the quiz section (.936)
# Got .94 when I switched the n_neighbors to 4, algorithm = 'auto'
# Tried different parameters, but the accuracy was below .936

neighbors = KNeighborsClassifier(n_neighbors = 2, algorithm='ball_tree')
neighbors = neighbors.fit(features_train, labels_train)
print 'prediction'
prediction = neighbors.predict(features_test)

print 'accuracy'
print accuracy_score(prediction, labels_test)




### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

'''
KNN is a classification (and also regression) supervised algorithm 
that uses K as the number of data points that are enclosed in a circle,
with the data point we are trying to classify right in the center.


'''








try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
