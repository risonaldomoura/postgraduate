#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 20:33:25 2024

@author: Risonaldo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% importar o dataset
df = pd.read_csv('dataset_tomate_com_severidade.csv')

X = df.drop(['id', 'Severidade'], axis = 1)
y = df['Severidade']

#%% separar dados de treinamento e dados de teste
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#%% padronizar os dados
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()  # média 0 e desvio padrão 1

X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

X_train_sc = pd.DataFrame(X_train_sc)
X_train_sc.columns = X_train.columns

X_test_sc = pd.DataFrame(X_test_sc)
X_test_sc.columns = X_train.columns


#%% Função para realizar seleção de features e validação cruzada
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score

def feature_selection_and_scoring(model, X_train, y_train, max_f=20):
    """
    Função para realizar a seleção de features e calcular a pontuação de R2.

    Parâmetros:
    - model: modelo de regressão a ser utilizado
    - X_train: dados de treino padronizados
    - y_train: variável dependente
    - max_f: número máximo de features a serem testadas
    
    Retorna:
    - scores: lista das médias das pontuações R2
    """
    scores = []
    for i in range(1, max_f + 1):
        selector = RFE(model, n_features_to_select=i, step=1)
        selector = selector.fit(X_train, y_train)
        X_sel = X_train[X_train.columns[selector.support_]]
        score = cross_val_score(model, X_sel, y_train, cv=10, scoring='r2')
        scores.append(np.mean(score))
    return scores

#%% Aplicação da função para diferentes modelos
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor

# Definição dos modelos
models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "ElasticNet": ElasticNet(),
    "Random Forest": RandomForestRegressor()
}

# Dicionário para armazenar os resultados de cada modelo
model_scores = {}

# Testando cada modelo
for name, model in models.items():
    scores = feature_selection_and_scoring(model, X_train_sc, y_train)
    model_scores[name] = scores

#%% gráficos comparativos
plt.figure(figsize=(10, 6))
for name, scores in model_scores.items():
    plt.plot(scores, label=name)
plt.legend()
plt.title('Comparação de R2 por número de features selecionadas')
plt.show()

#%% Impressão dos melhores resultados para cada modelo
print("Máximas pontuações e quantidade de features para cada modelo:")
for name, scores in model_scores.items():
    max_score = max(scores)
    best_features = scores.index(max_score) + 1  # índice +1 para considerar a quantidade de features
    print(f'{name}: Melhor R2 = {max_score:.4f} com {best_features} features')
    
#%% seleção de features final com o modelo de maior performance
# Exemplo com Random Forest, que obteve a maior performance
#modelo_rf = RandomForestRegressor()

modelo_rf = Ridge()

selector_rf = RFE(modelo_rf, n_features_to_select=15, step=1)

selector_rf = selector_rf.fit(X_train_sc, y_train)

sel_features_rf = X_train_sc.columns[selector_rf.support_]

X_sel_rf = X_train_sc[sel_features_rf]

#%% validação cruzada com Random Forest
score_rf = cross_val_score(modelo_rf, X_sel_rf, y_train, cv=10, scoring='r2')

print(f'Score médio R2 com Random Forest: {np.mean(score_rf)}')
print(f'Features selecionadas: {sel_features_rf}')

#%% modelo final - Random Forest
modelo_rf.fit(X_sel_rf, y_train)

#%% testar nos dados de teste
y_pred_rf = modelo_rf.predict(X_test_sc[sel_features_rf])

from sklearn.metrics import mean_squared_error, mean_absolute_error

r2_rf = modelo_rf.score(X_test_sc[sel_features_rf], y_test)
rmse_rf = (mean_squared_error(y_test, y_pred_rf) ** 0.5)
mae_rf = mean_absolute_error(y_test, y_pred_rf)

print('Random Forest - Teste:')
print(f'r2: {r2_rf}')
print(f'rmse: {rmse_rf}')
print(f'mae: {mae_rf}')
