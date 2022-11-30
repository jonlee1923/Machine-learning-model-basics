# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 19:41:22 2022

@author: Jonathan
"""

import numpy as np
import pandas as pd

training_data = pd.read_csv('storepurchasedata.csv')

training_data.describe()

#store independent variables in a numpy array, removes the last column
X = training_data.iloc[:, :-1].values

#gets the last column
y = training_data.iloc[:, -1].values

#split dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = .20, random_state=0)

#scale the dataa so that age and salary are in the same range so that the model is not influenced by the huge salaries
#the data is scaled such that the mean is 0 and SD is 1
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#build classification model
#n_neighbors is the number of neighbors
#p is the number of points
from sklearn.neighbors import KNeighborsClassifier
classifier= KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p =2)

#model training
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
y_prob = classifier.predict_proba(X_test)[:,1]


from sklearn.metrics import confusion_matrix
#used to predict accuracy of classification model

#TN = 3
#TP = 4
#FP = 0
#FN = 1
cm = confusion_matrix(y_test, y_pred)


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

#predict if a customer aged 40 w/ 20000 salary will buy or not
new_prediction = classifier.predict(sc.transform(np.array([[40, 20000]])))
new_prediction_proba = classifier.predict_proba(sc.transform(np.array([[40,20000]])))[:,1]

new_prediction2 = classifier.predict(sc.transform(np.array([[42, 50000]])))
new_prediction_proba2 = classifier.predict_proba(sc.transform(np.array([[42,50000]])))[:,1]


#pickle the model to serialize it/ convert to byte stream
# and then it can be deserialized in deployment
import pickle

#store the classifier object
model_file = "classifier.pickle"
pickle.dump(classifier, open(model_file, 'wb'))

#store scalar object
scaler_file = "sc.pickle"
pickle.dump(sc, open(scaler_file, 'wb'))