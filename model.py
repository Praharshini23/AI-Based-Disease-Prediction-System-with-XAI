import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

test = pd.read_csv()
train = pd.read_csv()

train = train.drop(columns=["Unnamed: 133"])

X_train = train.drop("prognosis", axis=1)
y_train = train["prognosis"]

X_test = test.drop("prognosis", axis=1)
y_test = test["prognosis"]

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

sample = X_train.iloc[0].copy()

prediction = model.predict([sample])

print("Predicted Disease:", prediction[0])

