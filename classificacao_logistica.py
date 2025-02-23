import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, weights):
    m = len(y)
    h = sigmoid(np.dot(X, weights))
    cost = (-1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return cost

def gradient_descent(X, y, weights, learning_rate, iterations):
    m = len(y)
    
    for _ in range(iterations):
        h = sigmoid(np.dot(X, weights))
        gradient = np.dot(X.T, (h - y)) / m
        weights -= learning_rate * gradient
    
    return weights

def train_logistic_regression(X, y, learning_rate=0.01, iterations=1000):
    X = np.insert(X, 0, 1, axis=1)  # Adiciona a coluna de bias
    weights = np.zeros(X.shape[1])  # Inicializa os pesos com zero
    weights = gradient_descent(X, y, weights, learning_rate, iterations)
    return weights

def predict(X, weights):
    X = np.insert(X, 0, 1, axis=1)  # Adiciona a coluna de bias
    probabilities = sigmoid(np.dot(X, weights))
    return [1 if p >= 0.5 else 0 for p in probabilities]

def compute_confusion_matrix(y_true, y_pred):
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    TP = np.sum((y_true == 1) & (y_pred == 1))
    
    confusion_mat = np.array([[TN, FP], [FN, TP]])
    labels = ["Desaprovado", "Aprovado"]

    # Exibindo a matriz de confusão com Matplotlib e Seaborn
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Previsto")
    plt.ylabel("Real")
    plt.title("Matriz de Confusão")
    plt.savefig("matriz_confusao_imagens/Matriz de confusao;learnig_rate: 0.01; iterations: 4000; threshold:0.5.png")
    plt.close()

    # Cálculo das métricas
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        "Acurácia": accuracy,
        "Precisão": precision,
        "Recall": recall,
        "F1-Score": f1_score
    }
    
    return confusion_mat, metrics

def classificacao_logistica(X_train, y_train, X_test, y_test):
    weights = train_logistic_regression(X_train, y_train, learning_rate=0.01, iterations=4000)
    predictions = predict(X_test, weights)

    predictions = np.array(predictions)

    confusion_mat, metrics = compute_confusion_matrix(y_test, predictions)
    
    #print("Pesos aprendidos:", weights)
    print("Matriz de confusão:", confusion_mat)
    print("Métricas:", metrics)