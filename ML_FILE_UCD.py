import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import category_encoders as ce

from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

titanic = pd.read_csv("C:/users/amcgrat/desktop/UCD PROGRAM/DATQCMP_UHC/titanic_train.csv")

print(pd.DataFrame(titanic).nunique())

titanic.drop(['passenger_id', 'name', 'ticket', 'cabin','boat','body','home.dest'], axis=1, inplace=True)

ports = pd.get_dummies(titanic.embarked, prefix='embarked')

titanic = titanic.join(ports)
titanic.drop(['embarked'], axis=1, inplace=True)

titanic.sex = titanic.sex.map({'male': 0, 'female': 1})

titanic.age.fillna(titanic.age.mean(), inplace=True)


y = titanic.survived.copy()
X = titanic.drop(['survived'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = pd.Series(model.predict(X_test))
y_test = y_test.reset_index(drop=True)

z = pd.concat([y_test, y_pred], axis=1)
z.columns = ['True', 'Prediction']
print(z.head(10))

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))


