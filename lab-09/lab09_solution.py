import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

loans = pd.read_csv('loan_data.csv')

print(loans.info())
print(loans.head())
print(loans.describe())

loans[loans['credit.policy'] == 1]['fico'].hist(bins=35, label='Credit Policy = 1')
loans[loans['credit.policy'] == 0]['fico'].hist(bins=35, label='Credit Policy = 0')
plt.legend()
plt.show()

loans[loans['not.fully.paid'] == 1]['fico'].hist(bins=35, label='Not Fully Paid = 1')
loans[loans['not.fully.paid'] == 0]['fico'].hist(bins=35, label='Not Fully Paid = 0')
plt.legend()
plt.show()

plt.figure(figsize=(11,6))
sns.countplot(x='purpose', hue='not.fully.paid', data=loans)
plt.xticks(rotation=45)
plt.show()

sns.jointplot(x='fico', y='int.rate', data=loans)
plt.show()

sns.lmplot(x='fico', y='int.rate', data=loans, hue='credit.policy', col='not.fully.paid')
plt.show()

cat_feats = ['purpose']
final_data = pd.get_dummies(loans, columns=cat_feats, drop_first=True)

print(final_data.info())

X = final_data.drop('not.fully.paid', axis=1)
y = final_data['not.fully.paid']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

dtree = DecisionTreeClassifier(random_state=101)
dtree.fit(X_train, y_train)

predictions = dtree.predict(X_test)

print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

rfc = RandomForestClassifier(n_estimators=600, random_state=101)
rfc.fit(X_train, y_train)

rfc_pred = rfc.predict(X_test)

print(classification_report(y_test, rfc_pred))
print(confusion_matrix(y_test, rfc_pred))

print('The Random Forest performed better overall in terms of accuracy, but it still struggled with predicting class 1.')
