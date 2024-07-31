from model import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

col_names = ["Gender", "Age", "EstimatedSalary", "Purchased"]
data = pd.read_csv("../data/data.csv", skiprows=1, header=None, names=col_names)
print(data.head())

X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values.reshape(-1, 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=41)

classifier = DecisionTreeClassifier(min_samples_split=3, max_depth=3)
classifier.fit(X_train, Y_train)
classifier.print_tree()

Y_pred = classifier.predict(X_test) 
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, Y_pred))