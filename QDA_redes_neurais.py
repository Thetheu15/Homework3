import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 🔹 Carregar os arquivos CSV
file_testing = "testing.csv"
file_training = "training.csv"
file_reduced_set = "reducedSet.csv"

df_testing = pd.read_csv(file_testing)
df_training = pd.read_csv(file_training)
df_reduced_set = pd.read_csv(file_reduced_set)

# 🔹 Extrair os nomes das colunas relevantes
selected_features = df_reduced_set["x"].values

# 🔹 Filtrar os dados
X_train_full = df_training[selected_features]
X_test = df_testing[selected_features]
y_train_full = df_training["Class"]
y_test = df_testing["Class"]

# 🔹 Codificar a variável alvo
label_encoder = LabelEncoder()
y_train_full = label_encoder.fit_transform(y_train_full)
y_test = label_encoder.transform(y_test)

# 🔹 Normalizar os dados
scaler = StandardScaler()
X_train_full = scaler.fit_transform(X_train_full)
X_test = scaler.transform(X_test)

# 🔹 Separar um conjunto de validação
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full)

# 🔹 Modelo QDA com validação cruzada
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)
y_pred_qda = qda.predict(X_test)

# 🔹 Avaliação do QDA
accuracy_qda = accuracy_score(y_test, y_pred_qda)
conf_matrix_qda = confusion_matrix(y_test, y_pred_qda)
print(f"\n🔹 QDA - Acurácia no Conjunto de Teste: {accuracy_qda:.4f}")
print("\nRelatório de Classificação (QDA):\n", classification_report(y_test, y_pred_qda))

# 🔹 Modelo de Redes Neurais
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 🔹 Early stopping para evitar overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 🔹 Treinar o modelo
history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=1)

# 🔹 Fazer previsões
y_pred_nn = (model.predict(X_test) > 0.5).astype("int32")

# 🔹 Avaliação do modelo de Redes Neurais
accuracy_nn = accuracy_score(y_test, y_pred_nn)
conf_matrix_nn = confusion_matrix(y_test, y_pred_nn)
print(f"\n🔹 Redes Neurais - Acurácia no Conjunto de Teste: {accuracy_nn:.4f}")
print("\nRelatório de Classificação (Redes Neurais):\n", classification_report(y_test, y_pred_nn))

# 🔹 Gerar gráficos
plt.figure(figsize=(12, 5))

# 🔹 Matriz de Confusão - QDA
plt.subplot(1, 2, 1)
sns.heatmap(conf_matrix_qda, annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de Confusão - QDA")
plt.xlabel("Previsto")
plt.ylabel("Real")

# 🔹 Matriz de Confusão - Redes Neurais
plt.subplot(1, 2, 2)
sns.heatmap(conf_matrix_nn, annot=True, fmt="d", cmap="Oranges")
plt.title("Matriz de Confusão - Redes Neurais")
plt.xlabel("Previsto")
plt.ylabel("Real")

plt.show()

# 🔹 Curvas de Treinamento
plt.figure(figsize=(12, 5))

# 🔹 Perda
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Treino", color="blue")
plt.plot(history.history["val_loss"], label="Validação", color="red")
plt.title("Perda Durante o Treinamento")
plt.xlabel("Época")
plt.ylabel("Perda")
plt.legend()

# 🔹 Acurácia
plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Treino", color="blue")
plt.plot(history.history["val_accuracy"], label="Validação", color="red")
plt.title("Acurácia Durante o Treinamento")
plt.xlabel("Época")
plt.ylabel("Acurácia")
plt.legend()

plt.show()

