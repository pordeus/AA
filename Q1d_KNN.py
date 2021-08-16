# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 18:21:13 2021

@author: daniel pordeus
Lista 02 - KNN
"""

import numpy as np
import heapq
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

breastcancer_dataset = np.genfromtxt(
    'E:\\Doutorado\\Aulas_Notas e Videos\\AA\\Listas\\Lista 2\\breastcancer.csv',
    delimiter=',', skip_header=1)

breastcancer = breastcancer_dataset.copy()

def normalizacao01_X(dados):
    dados_norm = np.zeros(dados.shape)
    zero = np.min(dados)
    um = np.max(dados)
    for x in range(dados.shape[0]):
        for y in range(dados.shape[1]):
            dados_norm[x,y] = (dados[x,y] - zero) / (um - zero)
    return dados_norm

#Distancia euclidiana entre dois pontos - professor
def distEuclidiana2(x, X):
    return -2 * x @ X.T + np.sum(x**2) + np.sum(X**2)


def minhaEuclidiana(x, X):
    distancia = []
    for i in range(len(X)):
        distancia.append((x - X[i])**2)
    return np.array(distancia)

def knn(knn, novo_x, X, Y):
    distX = minhaEuclidiana(novo_x, X)
    indices = heapq.nsmallest(knn, range(len(distX)), distX.take) #distX.argsort()[-3:][::-1] #retorna os indices dos knn menores
    #print(f"Indices: {indices}")
    classes = 0
    for i in range(knn):
        #print(f"Para i={i} indice={indices[i]} e Y={Y[indices[i]]}")
        classes =  classes + Y[indices[i]]
    if classes >= np.ceil(knn/2):
        return 1.0
    else:
        return 0.0
    
def avaliaClassificador(y_original, y_previsto):
    falsoPositivo = 0
    verdadeiroPositivo = 0
    falsoNegativo = 0
    verdadeiroNegativo = 0
    acuracia = 0
    for x in range(y_original.shape[0]):
        if y_original[x] == 0:
            if y_previsto[x] == 0:
                verdadeiroNegativo = verdadeiroNegativo + 1
            else:
                falsoNegativo = falsoNegativo + 1
        if y_original[x] == 1:
            if y_previsto[x] == 1:
                verdadeiroPositivo = verdadeiroPositivo + 1
            else:
                falsoPositivo = falsoPositivo + 1
    acuracia = np.mean(y_original == y_previsto)
    return acuracia, falsoPositivo, verdadeiroPositivo, falsoNegativo, verdadeiroNegativo

def formataSaida(valor):
    saidaFormatada = "{:.2f}".format(valor*100)
    return saidaFormatada + "%"

def F1_score(revocacao, precisao):
    return 2*(revocacao*precisao)/(revocacao+precisao)
   
## 
K = 10

tamanho_fold = int(np.ceil(len(breastcancer_dataset)/K))
fold = 0

teste = breastcancer[len(breastcancer)-tamanho_fold:,:]
treinamento = breastcancer[:-tamanho_fold,:]
fold = fold + tamanho_fold

y_treinamento = treinamento[:,30]
x_treinamento = treinamento[:, :29]
x_treinamento_norm = normalizacao01_X(treinamento[:, :29])

y_teste = teste[:,30]
x_teste = teste[:, :29]

y_teste_norm = y_teste

x_teste_norm = normalizacao01_X(teste[:, :29])


## Nova tentativa
y = breastcancer[:,30]
X = breastcancer[:, :29]
K = 10
k = 3
i_fold = 0
kf = KFold(n_splits=K, shuffle=True, random_state=5)
for train, valid in kf.split(X,y):
    i_fold = i_fold + 1
    x_treinamento_norm = X[train]#normalizacao01_X(X[train])
    y_treinamento = y[train]
    x_teste_norm = X[valid]#normalizacao01_X(X[valid])
    y_teste = y[valid]
    y_estimado = np.zeros(y_teste.shape)
    acuracia = 0
    i = 0
    for x in x_teste_norm:
        classe_estimada_x = knn(k,x,x_treinamento_norm, y_treinamento)
        classe_x = y_teste[i]
        y_estimado[i] = classe_x
        #print(f"Classe Estimada em {i}: {classe_estimada_x}. Classe Correta: {classe_x}")
        if classe_estimada_x == classe_x:
            acuracia = acuracia + 1
        i = i + 1
    print(f"Acuracia Rodada {i_fold}: {formataSaida(acuracia/len(x_teste_norm))}")
    acuracia, falsoPositivo, verdadeiroPositivo, falsoNegativo, verdadeiroNegativo = avaliaClassificador(y_teste, np.round(y_estimado))
    print(f"A acurácia no FOLD {i_fold} foi de {formataSaida(acuracia)}")
    #print(f"FP={falsoPositivo}, VP={verdadeiroPositivo}, FN={falsoNegativo}, VN={verdadeiroNegativo}")
    precisao = verdadeiroPositivo / (verdadeiroPositivo + falsoPositivo)
    format_precisao = "{:.2f}".format(precisao)
    revocacao = verdadeiroPositivo / (verdadeiroPositivo + falsoNegativo)
    format_revocacao = "{:.2f}".format(revocacao)
    print(f"Precisao={formataSaida(precisao)} Revocacao={formataSaida(revocacao)}")
    print(f"F1-Score={F1_score(revocacao, precisao)}")
    
    X_axis = np.arange(0, y_estimado.shape[0], dtype=int)
    Y_classe_1 = y_estimado
    
    fig2, ax2 = plt.subplots()
    
    ax2.plot(X_axis, Y_axis, 'x',color='tab:blue', label='Padrão')
    
    ax2.set_title(f"Verificação Rodada {i}")
    ax2.legend()
    plt.show()

    