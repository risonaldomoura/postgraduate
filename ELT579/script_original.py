# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 15:28:33 2022

@author: sarvi0
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier
from skopt.space import Categorical, Integer, Real

from lightgbm import LGBMClassifier

#baseline 76.555%
#modelo 3 76.794%


#%% abrir o datase de treino e teste

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


#%% pre-processamento dos dados

#descrição estátistica das features núméricas
est = train.describe()

print(train.info())

#verificar valores nulos ou NAN
print(train.isnull().sum())

print(test.isnull().sum())

#mapear as colunas
col = pd.Series(list(train.columns))

X_train = train.drop(['PassengerId', 'Survived'], axis = 1)

X_test = test.drop(['PassengerId'], axis = 1)

#%%
#criar feature

def criar_features(X):
  subs = {'female':1, 'male':0}
  X['mulher'] = X['Sex'].replace(subs)
  
  X['Fare'] = X['Fare'].fillna(X['Fare'].mean())
  
  X['Age'] = X['Age'].fillna(X['Age'].mean())
  
  X['Embarked'] = X['Embarked'].fillna('S')
  
  subs = {'S':1, 'C':2, 'Q':3}
  X['porto'] = X['Embarked'].replace(subs)
  
  X['crianca'] = 1
  X['crianca'] = np.where(X['Age'] < 12, 1, 0)
  
  return X

X_train = criar_features(X_train)
X_test = criar_features(X_test)

#%% Selecionar as features

features = ['Pclass', 'Age', 'SibSp',
       'Parch', 'Fare', 'mulher', 'porto', 'crianca']

X_train = X_train[features]
X_test = X_test[features]

y_train = train['Survived']


#%% Visualização

import matplotlib.pyplot as plt

for i in X_train.columns:
    plt.hist(X_train[i])
    plt.title(i)
    plt.show()
  
#%% Groupy

gp = train.groupby(['Survived']).count()

#%% pivot_table

table = pd.pivot_table(train, index = ['Survived'], columns = ['Pclass'], values = 'PassengerId', aggfunc = 'count')


#%% Padronização das variáveis

scaler = StandardScaler() #media 0 e desvio padrão 1

X_train_sc = scaler.fit_transform(X_train)

X_test_sc = scaler.transform(X_test)


#%% modelo e validação cruzada

#Logistic Regression
model_lr = LogisticRegression (random_state= 0 )

score = cross_val_score(model_lr, X_train_sc, y_train, cv = 10)

print(np.mean(score))

#%% Naive Bayes para Classificação

from sklearn.naive_bayes import GaussianNB

model_nb = GaussianNB()

score = cross_val_score(model_nb, X_train_sc, y_train, cv = 10)

print(np.mean(score))


#%% KNN para classificação
from sklearn.neighbors import KNeighborsClassifier

model_knn = KNeighborsClassifier(n_neighbors= 5, p = 2)

score = cross_val_score(model_knn, X_train_sc, y_train, cv = 10)

print(np.mean(score))


#%% SVM para classificação
from sklearn.svm import SVC

model_svc = SVC(C = 3, kernel = 'rbf', degree = 2, gamma = 0.1)

score = cross_val_score(model_svc, X_train_sc, y_train, cv = 10)

print(np.mean(score))


#%% Decision Tree

from sklearn.tree import DecisionTreeClassifier

model_dt = DecisionTreeClassifier(criterion = 'entropy', max_depth = 3, min_samples_split = 2, min_samples_leaf = 1, random_state = 0)

score = cross_val_score(model_dt, X_train_sc, y_train, cv = 10)

print(np.mean(score))


#%% Random Forest

from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier(criterion = 'entropy', n_estimators = 100, max_depth = 5, min_samples_split = 2, min_samples_leaf = 1, random_state = 0)

score = cross_val_score(model_rf, X_train_sc, y_train, cv = 10)

print(np.mean(score))


#%% Otimização de hiperparametros

from skopt import gp_minimize

def treinar_modelo(parametros):
  
  model_rf = RandomForestClassifier(criterion = parametros[0], n_estimators = parametros[1], max_depth = parametros[2], 
                                    min_samples_split = parametros[3], min_samples_leaf = parametros[4], random_state = 0, n_jobs = -1 )
  
  score = cross_val_score(model_rf, X_train_sc, y_train, cv = 10)
  
  mean_score = np.mean(score)
  
  print(np.mean(score))

  return -mean_score

parametros = [('entropy', 'gini'), 
              (100, 1000), 
              (3, 20),
              (2, 10),
              (1, 10)]


otimos = gp_minimize(treinar_modelo, parametros, random_state = 0, verbose = 1, n_calls = 30, n_random_starts = 10  )


print(otimos.fun, otimos.x)

#%% Ensanble model (Voting)
from sklearn.ensemble import VotingClassifier

model_voting = VotingClassifier(estimators = [('LR', model_lr), ('KNN', model_knn), ('SVC', model_svc), ('RF', model_rf)], voting = 'hard')

model_voting.fit(X_train_sc, y_train)

score = cross_val_score(model_voting, X_train_sc, y_train, cv = 10)

print(np.mean(score))

#%% modelo final

model_rf = RandomForestClassifier(criterion = otimos.x[0], n_estimators = otimos.x[1], max_depth = otimos.x[2], 
                                    min_samples_split = otimos.x[3], min_samples_leaf = otimos.x[4], random_state = 0, n_jobs = -1 )
  
model_rf.fit(X_train_sc, y_train)

y_pred = model_rf.predict(X_train_sc)

mc = confusion_matrix(y_train, y_pred) 
print(mc)

score = cross_val_score(model_rf, X_train_sc, y_train, cv = 10)

print(np.mean(score))

#%% predição nos dados de teste - voting

y_pred = model_voting.predict(X_test_sc)

submission = pd.DataFrame(test['PassengerId'])

submission['Survived'] = y_pred

submission.to_csv('UFV_ELT579_118515_Script_Original_Voting.csv', index = False)

#%% predição nos dados de teste - random forest

y_pred = model_rf.predict(X_test_sc)

submission = pd.DataFrame(test['PassengerId'])

submission['Survived'] = y_pred

submission.to_csv('UFV_ELT579_118515_Script_Original_RF.csv', index = False)



#%% 

# Modelo XGBoost

def treinar_modelo(parametros):
    
    model_xgb = XGBClassifier(learning_rate=parametros[0], 
                            n_estimators=parametros[1], 
                            max_depth=parametros[2],
                            min_child_weight=parametros[3], 
                            gamma=parametros[4], 
                            subsample=parametros[5], 
                            colsample_bytree=parametros[6],
                            random_state=0, n_jobs=-1, use_label_encoder=False, eval_metric='logloss')
    
    score = cross_val_score(model_xgb, X_train_sc, y_train, cv=10)
    mean_score = np.mean(score)
    return -mean_score

parametros = [
    Real(0.01, 0.3, prior='log-uniform'),  # learning_rate
    Integer(100, 1000),  # n_estimators
    Integer(3, 10),  # max_depth
    Integer(1, 10),  # min_child_weight
    Real(0, 0.5),  # gamma
    Real(0.6, 1.0),  # subsample
    Real(0.6, 1.0)  # colsample_bytree
]

# Otimização de hiperparâmetros
otimos = gp_minimize(treinar_modelo, parametros, random_state=0, verbose=1, n_calls=30, n_random_starts=10)

# Para exibir os melhores hiperparâmetros encontrados
print("Melhores parâmetros:", otimos.x)

#%% Criando o XGBoost com os parâmetros otimizados
model_xgb = XGBClassifier(learning_rate=otimos.x[0], 
                        n_estimators=otimos.x[1], 
                        max_depth=otimos.x[2], 
                        min_child_weight=otimos.x[3], 
                        gamma=otimos.x[4], 
                        subsample=otimos.x[5], 
                        colsample_bytree=otimos.x[6], 
                        random_state=0, n_jobs=-1, use_label_encoder=False, eval_metric='logloss')

model_xgb.fit(X_train_sc, y_train)

# Avaliando o desempenho com validação cruzada
score = cross_val_score(model_xgb, X_train_sc, y_train, cv=10)
print("Acurácia média após otimização:", np.mean(score))

#%% predição nos dados de teste - XGBoost

model_xgb.fit(X_train_sc, y_train)

y_pred = model_xgb.predict(X_test_sc)

submission = pd.DataFrame(test['PassengerId'])

submission['Survived'] = y_pred

submission.to_csv('UFV_ELT579_118515_Script_Original_Add_XGBoost.csv', index = False)

#%%

# Modelo LGBM

def treinar_modelo(parametros):
    
    model_lgbm = LGBMClassifier(learning_rate=parametros[0], 
                             n_estimators=parametros[1], 
                             max_depth=parametros[2],
                             num_leaves=parametros[3], 
                             min_child_samples=parametros[4], 
                             subsample=parametros[5], 
                             colsample_bytree=parametros[6],
                             random_state=0, n_jobs=-1)
    
    score = cross_val_score(model_lgbm, X_train_sc, y_train, cv=10)
    mean_score = np.mean(score)
    return -mean_score

# Espaço de busca para otimização de hiperparâmetros
parametros = [
    Real(0.01, 0.3, prior='log-uniform'),  # learning_rate
    Integer(100, 1000),  # n_estimators
    Integer(3, 10),  # max_depth
    Integer(20, 100),  # num_leaves
    Integer(10, 100),  # min_child_samples
    Real(0.6, 1.0),  # subsample
    Real(0.6, 1.0)  # colsample_bytree
]

# Otimização de hiperparâmetros
otimos = gp_minimize(treinar_modelo, parametros, random_state=0, verbose=1, n_calls=30, n_random_starts=10)

# Para exibir os melhores hiperparâmetros encontrados
print("Melhores parâmetros:", otimos.x)

#%% Criando o LightGBM com os parâmetros otimizados
model_lgbm = LGBMClassifier(learning_rate=otimos.x[0], 
                         n_estimators=otimos.x[1], 
                         max_depth=otimos.x[2], 
                         num_leaves=otimos.x[3], 
                         min_child_samples=otimos.x[4], 
                         subsample=otimos.x[5], 
                         colsample_bytree=otimos.x[6], 
                         random_state=0, n_jobs=-1)
model_lgbm.fit(X_train_sc, y_train)

# Avaliando o desempenho com validação cruzada
score = cross_val_score(model_lgbm, X_train_sc, y_train, cv=10)
print("Acurácia média após otimização:", np.mean(score))

#%% predição nos dados de teste - XGBoost

model_lgbm.fit(X_train_sc, y_train)

y_pred = model_lgbm.predict(X_test_sc)

submission = pd.DataFrame(test['PassengerId'])

submission['Survived'] = y_pred

submission.to_csv('UFV_ELT579_118515_Script_Original_Add_LGBM.csv', index = False)




















































