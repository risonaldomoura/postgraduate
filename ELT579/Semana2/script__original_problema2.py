# -*- coding: utf-8 -*-
"""

@author: sarvio valente

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#%% padronizar os dados

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler() #media 0 e desvio padrão 1

X_train_sc = scaler.fit_transform(X_train)

X_test_sc = scaler.transform(X_test)

X_train_sc = pd.DataFrame(X_train_sc)
X_train_sc.columns = X_train.columns

X_test_sc = pd.DataFrame(X_test_sc)
X_test_sc.columns = X_train.columns


#%% seleção de features
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

max_f = 20

lista_score = list()

for i in range(1, max_f +1):

  modelo_linear = LinearRegression()
  
  selector = RFE(modelo_linear, n_features_to_select = i, step = 1)
  
  selector = selector.fit(X_train_sc, y_train)
  
  mask = selector.support_
  
  features = X_train_sc.columns
  
  sel_features = features[mask]
  
  X_sel = X_train_sc[sel_features]
  
  score = cross_val_score(modelo_linear, X_sel, y_train, cv = 10, scoring = 'r2')
  
  print(np.mean(score))
  
  lista_score.append(np.mean(score))

#%% gráfico

import matplotlib.pyplot as plt

plt.plot(lista_score)

plt.show()

#%% seleção de features Final
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score


modelo_linear = LinearRegression()

selector = RFE(modelo_linear, n_features_to_select = 10, step = 1)

selector = selector.fit(X_train_sc, y_train)

mask = selector.support_

features = X_train_sc.columns

sel_features = features[mask]

X_sel = X_train_sc[sel_features]

score = cross_val_score(modelo_linear, X_sel, y_train, cv = 10, scoring = 'r2')

print(np.mean(score))
print(sel_features)

#%% validação cruzada

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

modelo_linear = LinearRegression()

score = cross_val_score(modelo_linear, X_sel, y_train, cv = 10, scoring = 'r2')

print(np.mean(score))


#%% modelo final - Regressão linear multipla

modelo_linear = LinearRegression()

modelo_linear.fit(X_sel, y_train)


#%% testar nos dados de teste

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

y_pred = modelo_linear.predict(X_test_sc[sel_features])

r2 = modelo_linear.score(X_test_sc[sel_features], y_test)

rmse = (mean_squared_error(y_test, y_pred)**0.5)

mae = mean_absolute_error(y_test, y_pred)

print('r2', r2)
print('rmse', rmse)
print('mae', mae)























































