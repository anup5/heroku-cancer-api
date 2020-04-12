#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

sns.set()

import pickle

# classification error metrics

import warnings

warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')

import matplotlib
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'

#get_ipython().run_line_magic('matplotlib', 'inline')


breast_cancer = load_breast_cancer()
X = pd.DataFrame(breast_cancer.data,columns=breast_cancer.feature_names)

X = X[['mean area', 'mean compactness']]
y = pd.Categorical.from_codes(breast_cancer.target,breast_cancer.target_names)
y = pd.get_dummies(y, drop_first=True)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=1)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)

# print("confusion_matrix:\n", confusion_matrix(y_test, y_pred))
# print("accuracy score : %.3f" % accuracy_score(y_test,y_pred))
# print("f1 score : %.3f" % f1_score(y_test,y_pred))


# # saving model to disk
pickle.dump(knn, open('model.pkl', 'wb'))


# # Loading model
model = pickle.load(open('model.pkl', 'rb'))

print(model.predict([[ 0.07856238, -0.63952448]]))
print(model.predict([[-0.91710572, -1.07260771]]))

