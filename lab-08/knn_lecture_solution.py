import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv('Classified Data', index_col=0)

X = df.drop('TARGET CLASS', axis=1)
y = df['TARGET CLASS']

scaler = StandardScaler()
scaled_features = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    scaled_features, y, test_size=0.30, random_state=101
)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)

print('K=1')
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

error_rate = []
for i in range(1, 40):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(X_train, y_train)
    pred_i = model.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

best_k = min(range(1, 40), key=lambda k: error_rate[k-1])
best_model = KNeighborsClassifier(n_neighbors=best_k)
best_model.fit(X_train, y_train)
best_pred = best_model.predict(X_test)

print(f'Best K from 1..39: {best_k}')
print(confusion_matrix(y_test, best_pred))
print(classification_report(y_test, best_pred))
