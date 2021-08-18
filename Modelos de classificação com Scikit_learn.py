# -*- coding: utf-8 -*-
"""
Created on Mon May  3 14:56:38 2021

@author: danie
"""

#Importando as bibliotecas
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

#Aqui vamos usar o arquivo de features onde usamos OHE (one hot encoding). Fique livre para escolher outro conjunto de features que geramos anteriormente ou mesmo gerar suas próprias features.
dfTitanic = pd.read_csv('lab5_train_no_nulls_no_outliers_ohe.csv')
dfTitanic.head(3)
dfTitanic = dfTitanic.sort_values('Survived')

# aqui montamos a matriz de atributos X e o vetor coluna de respostas Y.
# Note que não selecionamos algumas colnas, como Nome e Ticket
y = dfTitanic['Survived'].values
X = dfTitanic[['Age', 'SibSp', 'Parch', 'Fare', 'C', 'Q', 'S', '1', '2', '3', 'female', 'male']].values

# **TSNE**


X_tsne = TSNE(n_components=2).fit_transform(X)


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# We import seaborn to make nice plots.
import seaborn as sns

def scatter(x, colors):
    """this function plots the result
    - x is a two dimensional vector
    - colors is a code that tells how to color them: it corresponds to the target
    """
    
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 2))

    # We create a scatter plot.
    f = plt.figure(figsize=(10, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    
    ax.axis('off') # the axis will not be shown
    ax.axis('tight') # makes sure all data is shown
    
    # set title
    plt.title("Featurespace Visualization Titanic", fontsize=25)
    
    # legend with color patches
    survived_patch = mpatches.Patch(color=palette[1], label='Survived')
    died_patch = mpatches.Patch(color=palette[0], label='Died')
    plt.legend(handles=[survived_patch, died_patch], fontsize=20, loc=1)

    return f, ax, sc

# Use the data to draw ths scatter plot
scatter(X_tsne, y)

# Dividindo os dados em 5 folds.
kf = KFold(n_splits=5, shuffle=True, random_state=5)

# Assim como no laboratório anterior, aqui temos uma função para avaliar o desempenho dos algoritmos sem termos de ficar repetindo código. Ele retorna a média das métricas, acurácia no caso.
#Função idêntica à usada nos modelos de regressão.
def avalia_classificador(clf, kf, X, y, f_metrica):
    metrica_val = []
    metrica_train = []
    for train, valid in kf.split(X,y):
        x_train = X[train]
        y_train = y[train]
        x_valid = X[valid]
        y_valid = y[valid]
        clf.fit(x_train, y_train)
        y_pred_val = clf.predict(x_valid)
        y_pred_train = clf.predict(x_train)
        metrica_val.append(f_metrica(y_valid, y_pred_val))
        metrica_train.append(f_metrica(y_train, y_pred_train))
    return np.array(metrica_val).mean(), np.array(metrica_train).mean()

# Para simplificar a apresentação dos resultados e evitar repetição de código criamos a seguinte função auxuliar para imprimir os resultados.
def apresenta_metrica(nome_metrica, metrica_val, metrica_train, percentual = False):
    c = 100.0 if percentual else 1.0
    print('{} (validação): {}{}'.format(nome_metrica, metrica_val * c, '%' if percentual else ''))
    print('{} (treino): {}{}'.format(nome_metrica, metrica_train * c, '%' if percentual else ''))

## ADABOOST
    
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

best_acc = 0
for n in [50, 200, 500]:
    for l in [1, 0.5, 0.3, 0.2]:
        #print('n_estimators={}, learning_rate = {}'.format(n, l))
        ada = AdaBoostClassifier(n_estimators=1000, learning_rate=0.3)
        media_acuracia_val, media_acuracia_train = avalia_classificador(ada, kf, X, y, accuracy_score) 
        #apresenta_metrica('Acurácia', media_acuracia_val, media_acuracia_train, percentual=True)
        if media_acuracia_val > best_acc:
            best_acc = media_acuracia_val
            best_train = media_acuracia_train
            best_n = n
            best_l = l
print('melhor resultado: n_estimators={}, learning_rate = {}, acuraria = {}, treinamento={}'.format(best_n, best_l, best_acc, best_train))

ada = AdaBoostClassifier(n_estimators=1000, learning_rate=0.3)

media_acuracia_val, media_acuracia_train = avalia_classificador(ada, kf, X, y, accuracy_score) 
apresenta_metrica('Acurácia', media_acuracia_val, media_acuracia_train, percentual=True)

media_auc_val, media_auc_train = avalia_classificador(ada, kf, X, y, roc_auc_score) 
apresenta_metrica('AUC', media_auc_val, media_auc_train, percentual=True)


## Regressão logística

#Aqui usamos a classe LogisticRegression. Experimente variar os parâmetros do classificador.
# https://scikit-learn.org/0.17/modules/generated/sklearn.linear_model.LogisticRegression.html

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
lr = LogisticRegression(solver='liblinear')

media_acuracia_val, media_acuracia_train = avalia_classificador(lr, kf, X, y, accuracy_score) 
apresenta_metrica('Acurácia', media_acuracia_val, media_acuracia_train, percentual=True)

media_auc_val, media_auc_train = avalia_classificador(lr, kf, X, y, roc_auc_score) 
apresenta_metrica('AUC', media_auc_val, media_auc_train, percentual=True)

## SVM
#Aqui usamos a classe SVC. Experimente variar os parâmetros do classificador.

from sklearn import svm
svc = svm.SVC(gamma='auto')

media_acuracia_val, media_acuracia_train = avalia_classificador(svc, kf, X, y, accuracy_score) 
apresenta_metrica('Acurácia', media_acuracia_val, media_acuracia_train, percentual=True)

media_auc_val, media_auc_train = avalia_classificador(svc, kf, X, y, roc_auc_score) 
apresenta_metrica('AUC', media_auc_val, media_auc_train, percentual=True)

# KNN

# Aqui usamos a classe KNeighborsClassifier. Experimente variar os parâmetros do classificador.
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)

media_acuracia_val, media_acuracia_train = avalia_classificador(neigh, kf, X, y, accuracy_score) 
apresenta_metrica('Acurácia', media_acuracia_val, media_acuracia_train, percentual=True)

media_auc_val, media_auc_train = avalia_classificador(neigh, kf, X, y, roc_auc_score) 
apresenta_metrica('AUC', media_auc_val, media_auc_train, percentual=True)

#Arvore de Decisão

from sklearn import tree
dt = tree.DecisionTreeClassifier(max_depth=3)

media_acuracia_val, media_acuracia_train = avalia_classificador(dt, kf, X, y, accuracy_score) 
apresenta_metrica('Acurácia', media_acuracia_val, media_acuracia_train, percentual=True)

media_auc_val, media_auc_train = avalia_classificador(dt, kf, X, y, roc_auc_score) 
apresenta_metrica('AUC', media_auc_val, media_auc_train, percentual=True)

from sklearn import tree
import graphviz 
dot_data = tree.export_graphviz(dt, out_file=None, 
                                feature_names=['Age', 'SibSp', 'Parch', 'Fare', 'C', 'Q', 'S', '1', '2', '3', 'female', 'male'],  
                                filled=True, rounded=True,  
                                special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 










