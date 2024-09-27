#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 21:28:53 2024

@author: Risonaldo
"""

import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import confusion_matrix

#XGBoost
from xgboost import XGBClassifier
from skopt.space import Categorical, Integer, Real

#LGBM
from lightgbm import LGBMClassifier

#%% loading dataset
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

#%% Analisando dados totais

train_df['Survived'].value_counts(normalize=True)

#%%Analisando Feature Sex:

train_df['Sex'].value_counts().to_frame()

train_df.groupby('Sex').Survived.mean()


#%% Analisando Feature Pclass: 

train_df.groupby(['Pclass']).Survived.mean().to_frame()

train_df.groupby(['Pclass', "Sex"]).Survived.mean().to_frame()

#%% Analyze Feature Age

def plot_kernel_density_estimate_survivors(dataset, feature1, title, fsize = (5,5)):
    fig, ax = plt.subplots(figsize=fsize)
    ax.set_title(title) 
    sns.kdeplot(dataset[feature1].loc[train_df["Survived"] == 1],
                shade= True, ax=ax, label='Survived').set_xlabel(feature1)
    sns.kdeplot(dataset[feature1].loc[train_df["Survived"] == 0],
                shade=True, ax=ax, label="Died")

plot_kernel_density_estimate_survivors(train_df, "Age", "Distribuição gaussiana Sobreviventes vs. Mortes")

#%%Analisando Feature Age e Sex juntas

def plot_swarm_survivors(dataset, feature1, feature2, title, fize = (155)):
    fig, ax = plt.subplots(figsize=(18,5))
    # Turns off grid on the left Axis.
    ax.grid(True)
    #plt.xticks(list(range(0,100,2)))
    #plt.xticks(list(range(0,3,1)))
    plt.xticks(list(range(0,2,1)))
    sns.swarmplot(y=feature1, x=feature2, hue='Survived',data=train_df, dodge=True).set_title(title)

#%%Analisando Feature Age e Sex juntas
plot_swarm_survivors(train_df, "Sex", "Age", "Sobreviventes conforme idade e sexo")

    
#%% Analisando Features Age and Pclass juntas

plot_swarm_survivors(train_df, "Age", "Pclass", "Sobreviventes conforme idade e classe")

#%%Analisando Feature Fare

plot_distribution(train_df, "Fare", "Distribuição gaussiana de passageiros por Fare")

plot_swarm_survivors(train_df, "Fare", "Sex","Sobreviventes por Sexo e Fare")

len(train_df.loc[train_df.Fare==0])
#15


#%%
# Replace Fare == 0 with nan
train_df.loc[train_df['Fare'] == 0, 'Fare'] = np.NaN
test_df.loc[train_df['Fare'] == 0, 'Fare'] = np.NaN

#%% Analisando Feature Embarked

pd.pivot_table(train_df, index = 'Survived', columns = 'Embarked', values = 'Ticket' ,aggfunc ='count')

#%% Analisando Features Embarked e Pclass juntas

train_df.groupby(['Embarked', 'Pclass']).Survived.sum().to_frame()

#%%Analisando Features Embarked e Sex juntas

train_df.groupby(['Embarked', 'Sex']).Survived.sum().to_frame()

#%% Analisando Feature SibSp:

train_df.groupby(['SibSp']).Survived.mean().to_frame()

#%% Analisando Feature Parch

train_df.groupby(['Parch']).Survived.mean().to_frame()

    
#%% feature engineering

train_df['Title'] = train_df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
test_df['Title'] = test_df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())

#%%
train_df['Title'].replace(['Mme', 'Ms', 'Lady', 'Mlle', 'the Countess', 'Dona'], 'Miss', inplace=True)
test_df['Title'].replace(['Mme', 'Ms', 'Lady', 'Mlle', 'the Countess', 'Dona'], 'Miss', inplace=True)

train_df['Title'].replace(['Major', 'Col', 'Capt', 'Don', 'Sir', 'Jonkheer'], 'Mr', inplace=True)
test_df['Title'].replace(['Major', 'Col', 'Capt', 'Don', 'Sir', 'Jonkheer'], 'Mr', inplace=True)

#%%
# Extract Leading Letter:
train_df['Ticket_2letter'] = train_df.Ticket.apply(lambda x: x[:2])
test_df['Ticket_2letter'] = test_df.Ticket.apply(lambda x: x[:2])

# Extract Ticket Lenght:
train_df['Ticket_len'] = train_df.Ticket.apply(lambda x: len(x))
test_df['Ticket_len'] = test_df.Ticket.apply(lambda x: len(x))

# Extract Number of Cabins:
train_df['Cabin_num'] = train_df.Ticket.apply(lambda x: len(x.split()))
test_df['Cabin_num'] = test_df.Ticket.apply(lambda x: len(x.split()))

# Extract Leading Letter:
train_df['Cabin_1letter'] = train_df.Ticket.apply(lambda x: x[:1])
test_df['Cabin_1letter'] = test_df.Ticket.apply(lambda x: x[:1])

#%%
train_df['Fam_size'] = train_df['SibSp'] + train_df['Parch'] + 1
test_df['Fam_size'] = test_df['SibSp'] + test_df['Parch'] + 1

#%%
# Creation of four groups
train_df['Fam_type'] = pd.cut(train_df.Fam_size, [0,1,4,7,11], labels=['Solo', 'Small', 'Big', 'Very big'])
test_df['Fam_type'] = pd.cut(test_df.Fam_size, [0,1,4,7,11], labels=['Solo', 'Small', 'Big', 'Very big'])

#%%
features = ['Pclass', 'Fare', 'Title', 'Embarked', 'Fam_type', 'Ticket_len', 'Ticket_2letter']

X_train = train_df[features]
X_test = test_df[features]
y_train = train_df['Survived']

#%%

#categorical_cols = ['Fare', 'Pclass', 'Title', 'Embarked', 'Fam_type', 'Ticket_len', 'Ticket_2letter']

numerical_cols = ['Fare', 'Pclass']
categorical_cols = ['Title', 'Embarked', 'Fam_type', 'Ticket_len', 'Ticket_2letter']

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        #('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

#%%
# Bundle preprocessing and modeling code 
titanic_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(random_state=0, n_estimators=500, max_depth=5))
])

print('Cross validation score: {:.3f}'.format(cross_val_score(titanic_pipeline, X, y, cv=10).mean()))

#%%Otimizando os parâmetros do modelo RF

from skopt import gp_minimize

def treinar_modelo(parametros):
    
    pipeline_rf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestClassifier(
            criterion=parametros[0], n_estimators=parametros[1], 
            max_depth=parametros[2], min_samples_split=parametros[3],
            min_samples_leaf=parametros[4], random_state=0, n_jobs=-1
            )
        )
    ])

    score = cross_val_score(pipeline_rf, X_train, y_train , cv = 10)
    mean_score = np.mean(score)
    return -mean_score

parametros = [('entropy', 'gini'), 
              (100, 1000),
              (3, 20),
              (2, 10),
              (1, 10)]

otimos = gp_minimize(
    treinar_modelo, parametros, 
    random_state=0, verbose = 1, 
    n_calls = 30, n_random_starts = 10
)

print(otimos.fun, otimos.x)


#%% criando um RF com os parâmetros encontrados na otimização

pipeline_rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(
        criterion=otimos.x[0], n_estimators=otimos.x[1], 
        max_depth=otimos.x[2], min_samples_split=otimos.x[3], 
        min_samples_leaf=otimos.x[4], random_state=0, n_jobs=-1
        )
    )
])

score = cross_val_score(pipeline_rf, X_train , y_train , cv=10)
print(np.mean(score))

#%% Training and submission

# Training
pipeline_rf.fit(X_train, y_train)

predictions = pipeline_rf.predict(X_test)

output = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': predictions})
output.to_csv('UFV_ELT579_118515_New_Features_RF_Only_Categorical.csv', index=False)
print('Your submission was successfully saved!')
















