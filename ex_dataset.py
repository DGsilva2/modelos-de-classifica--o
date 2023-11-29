import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1)

mnist.keys()

x, y= mnist['data'].values, mnist['target'].values
y = y.astype(np.float) #convertando o Y para float

x
y

n = 4
plt.imshow(x[n].reshape(28,28), cmap='binary')
print(y[n])

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

y_train_5 = (y_train == 5)

pd.Series(y_train_5).value_counts()

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier()
sgd_clf.fit(x_train, y_train_5)

n = 10
plt.imshow(x_train[n].reshape(28,28))

print('Classe real: ',y_train_5[n])
print('Classe predita pelo modelo: ',sgd_clf.predict([x_train[n]]))

#medindo a acuracia de um modelo binario

from sklearn.model_selection import cross_val_score

cross_val_score(sgd_clf, x_train, y_train_5, cv=3, scoring='accuracy')

#natrix de confusao
from sklearn.metrics import confusion_matrix

confusion_matrix(y_train_5, sgd_clf.predict(x_train))

from sklearn.metrics import precision_score, recall_score

y_train_pred = sgd_clf.predict(x_train)

print('Precision: ', precision_score(y_train_5, y_train_pred))
print('Recall: ', recall_score(y_train_5, y_train_pred))

from sklearn.metrics import classification_report

print(classification_report(y_train_5, y_train_pred))

sgd_clf.fit(x_train, y_train)

n = 5
digit = x_train[n]
plt.imshow(digit.reshape(28,28))

sgd_clf.predict([digit])

from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, x_train, y_train, cv=3)

conf_max = confusion_matrix(y_train, y_train_pred)
print(conf_max)
print(classification_report(y_train, y_train_pred))

fig, ax= plt.subplots(figsize= (25,8))
sns.heatmap(conf_max, annot=True, fmt=".0f")

row_sums = conf_max.sum(axis=1, keepdims=True)
norm_conf_mx = conf_max / row_sums

np.fill_diagonal(norm_conf_mx, 0)
fig, ax= plt.subplots(figsize= (30,10))
sns.heatmap(norm_conf_mx, ax=ax, annot=True)


#multilabel
from sklearn.neighbors import KNeighborsClassifier

y_train_large= (y_train >= 7)
y_train_odd = (y_train % 2 ==1)
y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(x_train, y_multilabel)

n= 4 
digit = x_train[n]
plt.imshow(digit.reshape(28,28))

knn_clf.predict([digit])