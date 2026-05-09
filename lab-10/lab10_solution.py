import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

iris = sns.load_dataset('iris')

sns.pairplot(iris,hue='species',palette='Dark2')
plt.show()

setosa = iris[iris['species'] == 'setosa']
sns.kdeplot(data=setosa,x='sepal_width',y='sepal_length',cmap='plasma',fill=True,thresh=0.05)
plt.show()

X = iris.drop('species',axis=1)
y = iris['species']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=101)

model = SVC()
model.fit(X_train,y_train)

predictions = model.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

param_grid = {'C': [0.1,1,10,100], 'gamma': [1,0.1,0.01,0.001], 'kernel': ['rbf']}

grid = GridSearchCV(SVC(),param_grid,verbose=3,cv=3)
grid.fit(X_train,y_train)
print(grid.best_params_)

grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test,grid_predictions))
print(classification_report(y_test,grid_predictions))
