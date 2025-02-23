from tratamento_dados import tratamento_dados
from classificacao_logistica import classificacao_logistica


X_train, y_train, X_val, y_val, X_test, y_test = tratamento_dados()

classificacao_logistica(X_train, y_train, X_test, y_test)
