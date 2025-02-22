import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split

def tratamento_dados():
    testing = pd.read_csv("data/testing.csv")
    training = pd.read_csv("data/training.csv")
    reducedSet = pd.read_csv("data/reducedSet.csv")

    X_train = training.drop(columns=['Class'])
    y_train = training['Class']

    X_test = testing.drop(columns=['Class'])
    y_test = testing['Class']

    X_train = X_train[X_train.columns.intersection(reducedSet.iloc[:, 0])]
    X_test = X_test[X_test.columns.intersection(reducedSet.iloc[:, 0])]

    X_train, _ ,y_train, _ = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    X_test, _ ,y_test, _ = train_test_split(X_test, y_test, test_size=0.2, random_state=42, stratify=y_test)

    return X_train, y_train, X_test, y_test