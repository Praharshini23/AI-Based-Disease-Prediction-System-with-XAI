import pandas as pd

test = pd.read_csv()
train = pd.read_csv()

train = train.drop(columns=["Unnamed: 133"])

print("Training Shape:", train.shape)
print("Testing Shape:", test.shape)

print("\nColumns:")
print(train.columns)

print("\nMissing Values:")
print(train.isnull().sum().sum())

print("\nUnique Diseases:")
print(train["prognosis"].nunique())

print("\nDisease Names:")
print(train["prognosis"].unique())


print("Training Shape:", train.shape)
print("Testing Shape:", test.shape)

print("\nNumber of Symptoms:")
print(len(train.columns) - 1)

print("\nNumber of Diseases:")
print(train["prognosis"].nunique())
