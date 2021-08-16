# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 20:44:05 2021

@author: daniel pordeus

Lista 2 - Análise do Discriminante Gaussiano
"""

import numpy as np
#import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

breastcancer_dataset = np.genfromtxt(
    'E:\\Doutorado\\Aulas_Notas e Videos\\AA\\Listas\\Lista 2\\breastcancer.csv',
    delimiter=',', skip_header=1)

breastcancer = breastcancer_dataset.copy()

y = breastcancer[:,30]
X = breastcancer[:, :29]

#função de avaliação
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

#Separação das classes
nonzeros = np.nonzero(y)
Classe_1 = np.zeros((len(nonzeros[0]), X.shape[1]))
index = 0
for i in nonzeros[0]:
    Classe_1[index] = X[i]
    index = index + 1

# Valores pré-obtidos para aceleração dos cálculos - valores repetitivos
Matriz_Cov_Classe_1 = np.cov(Classe_1.T)
Inversa_Cov_1 = np.linalg.inv(np.cov(Matriz_Cov_Classe_1))
Log_Matriz_Cov_Classe_1  = np.log(Matriz_Cov_Classe_1)

zeros = np.argwhere(y==0)
Classe_0 = np.zeros((zeros.shape[0], X.shape[1]))
index = 0
for j in zeros:
    #print(X[j])
    Classe_0[index] = X[j]
    index = index + 1

# Valores pré-obtidos para aceleração dos cálculos - valores repetitivos
Matriz_Cov_Classe_0 = np.cov(Classe_0.T)
Inversa_Cov_0 = np.linalg.inv(np.cov(Matriz_Cov_Classe_0))
Log_Matriz_Cov_Classe_0  = np.log(Matriz_Cov_Classe_0)

## Probabilidade das Classes
P_Classe_1 = len(Classe_1)/len(y)
Log_Classe_1 = np.log(P_Classe_1)
P_Classe_0 = len(Classe_0)/len(y) # OU 1 - P_Classe_1 
Log_Classe_0 = np.log(P_Classe_0)
#Medias
Media_Classe_1 = np.mean(Classe_1)
Media_Classe_0 = np.mean(Classe_0)

# Inicio dos Folds
K = 10
kf = KFold(n_splits=K, shuffle=True, random_state=5)
i_fold = 0
acuraciaFinal = 0
f1Final = 0
precisaoFinal = 0
revocacaoFinal = 0
for train, valid in kf.split(X,y):
    #x_treinamento_norm = X[train] #normalizacao01_X(X[train])
    #y_treinamento = y[train]
    x_teste_norm = X[valid] #normalizacao01_X(X[valid])
    y_teste = y[valid]
 
    ## INICIO ##
    i = 0
    Classe_predita = np.zeros(y_teste.shape)
    for x in x_teste_norm:
        #print(x)
        Classe_Predita_1 = -0.5 * Log_Matriz_Cov_Classe_1  
        - 0.5 * (x - Media_Classe_1).T @ Inversa_Cov_1 @ (x - Media_Classe_1)
        + Log_Classe_1
        
        Classe_Predita_0 = -0.5 * Log_Matriz_Cov_Classe_0 
        - 0.5 * (x - Media_Classe_0).T @ Inversa_Cov_0 @ (x - Media_Classe_0)
        + Log_Classe_0
        
        P_Ck_1 = Classe_Predita_1 * P_Classe_1
        P_Ck_0 = Classe_Predita_0 * P_Classe_0
        
        Classe = 0
        if np.nanmean(P_Ck_1) > np.nanmean(P_Ck_0):
            Classe = 1
        
        Classe_predita[i] = Classe
        i = i + 1
    i_fold = i_fold + 1    
    acuracia, falsoPositivo, verdadeiroPositivo, falsoNegativo, verdadeiroNegativo = avaliaClassificador(y_teste, Classe_predita)
    print(f"A acurácia no FOLD {i_fold} foi de {formataSaida(acuracia)}")
    #print(f"FP={falsoPositivo}, VP={verdadeiroPositivo}, FN={falsoNegativo}, VN={verdadeiroNegativo}")
    try:
        precisao = verdadeiroPositivo / (verdadeiroPositivo + falsoPositivo)
        format_precisao = "{:.2f}".format(precisao)
        revocacao = verdadeiroPositivo / (verdadeiroPositivo + falsoNegativo)
        format_revocacao = "{:.2f}".format(revocacao)
        f1 = F1_score(revocacao, precisao)
        format_f1 = "{:.2f}".format(f1)
        print(f"Precisao={formataSaida(precisao)} Revocacao={formataSaida(revocacao)}")
        print(f"F1-Score={format_f1}")
        acuraciaFinal = acuraciaFinal + acuracia
        f1Final =  f1Final + f1
        precisaoFinal = precisaoFinal + precisao
        revocacaoFinal = revocacaoFinal + revocacao
    except :
        print(f"VP={verdadeiroPositivo} FP={falsoPositivo} VN={verdadeiroNegativo} FN={falsoNegativo}")

acuraciaFinal = acuraciaFinal / K
f1Final =  f1Final / K
precisaoFinal = precisaoFinal / K
revocacaoFinal = revocacaoFinal / K
f1Final = "{:.2f}".format(f1Final)

print(f"Acuracia Geral= {formataSaida(acuraciaFinal)}")
print(f"F1 Score Geral= {f1Final}")
print(f"Precisão Geral= {formataSaida(precisaoFinal)}")
print(f"Revocação Geral= {formataSaida(revocacaoFinal)}")








