import pandas as pd
from sklearn.model_selection import train_test_split
from IPython.display import display
import numpy as np

from sklearn.preprocessing import StandardScaler

def tratamento_dados():
    testing = pd.read_csv("data/testing.csv")
    training = pd.read_csv("data/training.csv")
    reducedSet = pd.read_csv("data/reducedSet.csv")

    # Converte a variável-alvo para 0 e 1
    training['Class'] = training['Class'].map({'successful': 1, 'unsuccessful': 0})
    testing['Class'] = testing['Class'].map({'successful': 1, 'unsuccessful': 0})

    # Separa variáveis de entrada (X) e saída (y)
    X_train = training.drop(columns=['Class'])
    y_train = training['Class'].values
    X_test = testing.drop(columns=['Class'])
    y_test = testing['Class'].values
    
    # Filtra as features selecionadas no reducedSet
    selected_features = reducedSet.iloc[:, 0].values  # Garante que pega os nomes das colunas
    X_train = X_train[X_train.columns.intersection(selected_features)]
    X_test = X_test[X_test.columns.intersection(selected_features)]

    # Divide apenas o conjunto de treino
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    """
    print("X_train:\n")
    display(X_train)
    print("y_train:\n")
    display(y_train)
    print("X_test:\n")
    display(X_test)
    print("y_test:\n")
    display(y_test)
    print(np.unique(y_train, return_counts=True))
    print(np.unique(y_test, return_counts=True))
    """

    return X_train, y_train, X_val, y_val, X_test, y_test
