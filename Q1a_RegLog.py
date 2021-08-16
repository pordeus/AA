# -*- coding: utf-8 -*-
"""
Created on Fri May 28 17:23:02 2021

@author: daniel pordeus

Lista 02 - Q01a
Regressão Logistica com K-Fold
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


breastcancer_dataset = np.genfromtxt(
    'E:\\Doutorado\\Aulas_Notas e Videos\\AA\\Listas\\Lista 2\\breastcancer.csv',
    delimiter=',', skip_header=1)

breastcancer = breastcancer_dataset.copy()

## Variáveis globais para todos os casos
num_epocas = 2000
alpha =  0.005
N = breastcancer.shape[0]

#np.random.shuffle(breastcancer)
    
def normalizacao01_X(dados):
    dados_norm = np.zeros(dados.shape)
    zero = np.min(dados)
    um = np.max(dados)
    for x in range(dados.shape[0]):
        for y in range(dados.shape[1]):
            dados_norm[x,y] = (dados[x,y] - zero) / (um - zero)
    return dados_norm

def funcaoFi(z):
    return 1 / (1 + np.e**(-z))

def calculaErro(y_original, y_calibrado):
    erro = 0
    for x in range(y_original.shape[0]):
        if y_original[x] != y_calibrado[x]:
            erro = erro + 1
    return erro

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

## Nova tentativa
y = breastcancer[:,30]
X = breastcancer[:, :29]
K = 10
num_epocas = 3000
Custo = np.zeros((K,num_epocas))
Custo_fold = np.zeros(K)
i_fold = 0

kf = KFold(n_splits=K, shuffle=True, random_state=5)
for train, valid in kf.split(X,y):
    x_treinamento_norm = normalizacao01_X(X[train])
    y_treinamento = y[train]
    x_teste_norm = normalizacao01_X(X[valid])
    y_teste = y[valid]
    ## INICIO ##
    W = np.random.rand(x_treinamento_norm.shape[1])
    epoca = 0
    
    Custo_teste = []
     
    while epoca < num_epocas:
        #Custo.append(0)
        Custo_teste.append(0) 
        Y = funcaoFi(x_treinamento_norm @ W)
        
        erros = y_treinamento - Y
            
        Custo[i_fold, epoca] = -1 / N * np.sum(erros @ x_treinamento_norm)
        
        pred = x_teste_norm @ W
        
        erro_teste = y_teste - pred
        
        Custo_teste[epoca] = -1 / N * np.sum(erro_teste @ x_teste_norm)
        Custo_fold = Custo_fold + Custo_teste[epoca]
        #print(f"Custo Fold={Custo_fold} epoca={epoca} Fold={fold}")
        W = W + (alpha * (np.sum(x_treinamento_norm.T @ erros) / x_treinamento_norm.shape[0])) 
            
        epoca = epoca + 1
        #fim do while
    print(f"ERRO Fold={i_fold+1} é {np.sum(Custo_teste)}")
    acuraciaTreino = 1 - calculaErro(y_treinamento,np.round(Y))/y_treinamento.shape[0]
    acuraciaTeste = 1 - calculaErro(y_teste, np.round(pred))/y_teste.shape[0]
    acuracia, falsoPositivo, verdadeiroPositivo, falsoNegativo, verdadeiroNegativo = avaliaClassificador(y_teste, np.round(pred))
    print(f"A acurácia no FOLD {i_fold+1} foi de {formataSaida(acuracia)}")
    #print(f"FP={falsoPositivo}, VP={verdadeiroPositivo}, FN={falsoNegativo}, VN={verdadeiroNegativo}")
    try:
        precisao = verdadeiroPositivo / (verdadeiroPositivo + falsoPositivo)
        format_precisao = "{:.2f}".format(precisao)
        revocacao = verdadeiroPositivo / (verdadeiroPositivo + falsoNegativo)
        format_revocacao = "{:.2f}".format(revocacao)
        print(f"Precisao={formataSaida(precisao)} Revocacao={formataSaida(revocacao)}")
        print(f"F1-Score={F1_score(revocacao, precisao)}")
    except :
        print(f"VP={verdadeiroPositivo} FP={falsoPositivo} VN={verdadeiroNegativo} FN={falsoNegativo}")
    print(f"Acurária Treino em Fold={i_fold+1} é {formataSaida(acuraciaTreino)}")
    print(f"Acurária Teste em Fold={i_fold+1} é {formataSaida(acuraciaTeste)}")
    Custo_fold[i_fold] = np.sum(Custo_teste)
    i_fold = i_fold + 1
     

Erro_fold = np.sum(Custo_fold)/N
print(f"Erro(fold)={Erro_fold}")
fig2, ax2 = plt.subplots(figsize=(12,8))
ax2.plot(np.arange(1,num_epocas+1), Custo[0], color='tab:blue', label='Teste Fold 1')
ax2.plot(np.arange(1,num_epocas+1), Custo[1], color='tab:green', label='Teste Fold 2')
ax2.plot(np.arange(1,num_epocas+1), Custo[2], color='tab:orange', label='Teste Fold 3')
ax2.plot(np.arange(1,num_epocas+1), Custo[3], color='tab:green', label='Teste Fold 4')
ax2.plot(np.arange(1,num_epocas+1), Custo[4], color='tab:gray', label='Teste Fold 5')
ax2.plot(np.arange(1,num_epocas+1), Custo[5], color='tab:brown', label='Teste Fold 6')
ax2.plot(np.arange(1,num_epocas+1), Custo[6], color='tab:purple', label='Teste Fold 7')
ax2.plot(np.arange(1,num_epocas+1), Custo[7], color='tab:olive', label='Teste Fold 8')
ax2.plot(np.arange(1,num_epocas+1), Custo[8], color='tab:pink', label='Teste Fold 9')
ax2.plot(np.arange(1,num_epocas+1), Custo[9], color='tab:cyan', label='Teste Fold 10')
ax2.plot(np.arange(1,num_epocas+1), Custo_teste, color='tab:red', label='Teste')
ax2.set_title("CUSTO")
ax2.legend()
plt.show()
