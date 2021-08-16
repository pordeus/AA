# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 22:05:52 2021

@author: daniel pordeus
NN
"""

#import random
import numpy as np
import math
import random
from matplotlib import pyplot as plt

# Funcoes uteis
def corAleatoria():
    red = random.random()
    green = random.random()
    blue = random.random()
    return (red, green, blue)

def norm_tanh(dados):
    dados_norm = np.zeros(dados.shape)
    for x in range(dados.shape[0]):
        mean = np.mean(dados[x])
        std = np.std(dados[x])
        for y in range(dados.shape[1]):
            dados_norm[x,y] = 0.5 * (np.tanh(0.01 * (dados[x,y] - mean)/std) + 1)
    return dados_norm

#Normalizacao [0,1] para matriz [n x 2]
def normalizacao01_X(dados):
    dados_norm = np.zeros(dados.shape)
    for x in range(dados.shape[0]):
        zero = np.min(dados[x])
        um = np.max(dados[x])
        for y in range(dados.shape[1]):
            dados_norm[x,y] = (dados[x,y] - zero) / (um - zero)
    return dados_norm

#Normalizacao [0,1] para vetores unidimensionais
def normalizacao01_Y(dados):
    dados_norm = np.zeros(dados.shape)
    zero = np.min(dados)
    um = np.max(dados)
    for x in range(dados.shape[0]):
        dados_norm[x] = (dados[x] - zero) / (um - zero)
    return dados_norm

#Normalizacao [-1,1] para matriz [n x 2]
def normalizacao11_X(dados):
    dados_norm = np.zeros(dados.shape)
    for x in range(dados.shape[0]):
        menos_um = np.min(dados[x])
        um = np.max(dados[x])
        media = (um + menos_um) /2
        for y in range(dados.shape[1]):
            dados_norm[x,y] = (dados[x,y] - media) / media
    return dados_norm

#Normalizacao [-1,1] para vetores unidimensionais
def normalizacao11_Y(dados):
    dados_norm = np.zeros(dados.shape)
    menos_um = np.min(dados)
    um = np.max(dados)
    media = (um + menos_um) /2
    for x in range(dados.shape[0]):
        dados_norm[x] = (dados[x] - media) / media
    return dados_norm

def matrizReLU(dados):
    matrizSaida = dados.copy()
    for i in range(dados.shape[0]):
        for j in range(dados.shape[1]):
            matrizSaida[i,j] = max(0,dados[i,j])
    return matrizSaida

def vetorReLU(dados):
    matrizSaida = dados.copy()
    for i in range(dados.shape[0]):
        matrizSaida[i] = max(0,dados[i])
    return matrizSaida

def relu(x):
    re = np.vectorize(lambda y: max(0, y))
    return re(x)

def softPlus(x):
    sp = np.vectorize(lambda y: math.log(1 + math.exp(y)))
    return sp(x)

def sigmoid(x):
    sig = np.vectorize(lambda y:  (1 - 1 / (1 + math.exp(y))) if y < 0 else  (1 / (1 + math.exp(-y))))
    return sig(x)

def ativTanH(dados):
    matrizSaida = dados.copy()
    for i in range(dados.shape[0]):
        for j in range(dados.shape[1]):
            matrizSaida[i,j] = 1 - np.tanh(dados[i,j])
    return matrizSaida

def derivada(x, function):
        if function == "sigmoid":
            return np.multiply(x, (1-x))
        elif function == "soft_plus":
            return sigmoid(x)
        elif function == "relu":
            d_relu = np.vectorize(lambda y: 0 if y < 0 else 1)
            return d_relu(x)

# carregamento dados de entrada
dataset = np.genfromtxt('E:\\Doutorado\\Aulas_Notas e Videos\\AA\\Listas\\Lista 3\\concrete.csv', delimiter=',', skip_header=1)
coluna = dataset.shape[1] - 1
#embaralhando entrada
np.random.shuffle(dataset)

#divisao de conjuntos treinamento, testee e validacao
treino_size = int(np.floor((0.6 * len(dataset))))
teste_size = int(np.floor((0.2 * len(dataset))))+1
valid_size = len(dataset) - int(np.floor((0.8 * len(dataset))))

treino_set = dataset[0:treino_size,:]
teste_set = dataset[treino_size:(treino_size+teste_size),:]
valid_set = dataset[(treino_size+teste_size):,:]

#Normalizacao
treino_norm_X = normalizacao11_X(treino_set[:,:coluna])
teste_set_X = normalizacao11_X(teste_set[:,:coluna])
valid_set_X = normalizacao11_X(valid_set[:,:coluna])

# Adição da Coluna 1
treino_norm_X = np.c_[np.ones(treino_norm_X.shape[0]), treino_norm_X]
teste_set_X = np.c_[np.ones(teste_set_X.shape[0]), teste_set_X]
valid_set_X = np.c_[np.ones(valid_set_X.shape[0]), valid_set_X]

#Y's
Y_treino = normalizacao11_Y(treino_set[:,coluna])
Y_teste = normalizacao11_Y(teste_set[:,coluna])
Y_valid = normalizacao11_Y(valid_set[:,coluna])

#Hiperparametros
alphas =  [0.01, 0.001, 0.0001] #np.random.random() # taxa de aprendizagem
hidden = [8, 16, 32]
melhor_combinacao = [1000, [], []] # [erro, Matriz de Pesos W, M]
#
epocas = 300
print("Inicio de Treinamento")
print("Erros - Grid Search")
fig2, ax2 = plt.subplots()
fig2, ax2 = plt.subplots(figsize=(12,8))
ax2.set_title('MSE x RMSE x MAE x MRE')
for alpha in alphas:
    print(f"Passo de Aprendizagem = {alpha}")
    for h in hidden:
        # Pesos - aleatorio de 0 a 1 - criação
        print(f"Camadas Ocultas = {h}")
        print(" # # # # # # ")
        W = np.random.rand(dataset.shape[1], h)
        M = np.random.rand((h+1))
        MSE = np.zeros(epocas)
        MAE = np.zeros(epocas)
        MRE = np.zeros(epocas)
        qtdEpocas = 0
        N = len(treino_norm_X)
        while qtdEpocas < epocas:
            
            Z = matrizReLU(treino_norm_X @ W)
            
            Z = np.c_[np.ones(Z.shape[0]), Z]
            
            O = softPlus(Z @ M) #ativTanH(Z @ M)
            
            Erro = Y_treino - O
            MSE[qtdEpocas] = np.sum(Erro)
            MRE[qtdEpocas] = np.sum(Erro / Y_treino)
            
            #Backprop
            delta = Erro * derivada(Z @ M,"soft_plus")
            part = delta.reshape(delta.shape[0], 1) @ M[1:].reshape(1, M.shape[0]-1)
            Chi = derivada(treino_norm_X @ W,"relu") * part 
            
            M = M + alpha * (delta @ Z)
            W = W + alpha * treino_norm_X.T @ Chi
            qtdEpocas += 1
        MRE = MRE / N
        MAE = MSE / N
        MSE = MSE**2 / N
        print(f"Erro = {np.sum(Erro)}  -- MSE = {np.sum(MSE)}")
        #Fim do Treinamento
        
        #desenhar grafico Erro / MSE
        
        # Verificação com Validação
        
        Z_valid = matrizReLU(valid_set_X @ W)
        Z_valid = np.c_[np.ones(Z_valid.shape[0]), Z_valid]
        
        O_valid = softPlus(Z_valid @ M) #ativTanH(Z @ M)
        
        Erro_valid = (Y_valid - O_valid)
        MSE_valid = Erro_valid**2 / len(valid_set_X)
        print(f"Erro Validação = {np.sum(Erro_valid)}")
        if np.sum(Erro_valid) < melhor_combinacao[0]:
            melhor_combinacao = [np.sum(Erro_valid), W, M] #armazena o melhor W
        
        #Montando gráfico
        ax2.plot(np.arange(epocas), MSE, color=corAleatoria(), label=f"MSE Alpha={alpha} Nh={h}")
        ax2.plot(np.arange(epocas), np.sqrt(MSE), color=corAleatoria(), label=f"RMSE Alpha={alpha} Nh={h}")
        ax2.plot(np.arange(epocas), MAE, color=corAleatoria(), label=f"MAE Alpha={alpha} Nh={h}")
        ax2.plot(np.arange(epocas), MRE, color=corAleatoria(), label=f"MRE Alpha={alpha} Nh={h}")
ax2.set_ylim([0, 750])
ax2.legend()
plt.show()

# teste
#melhor_combinacao[1] é o melhor W no Grid Search 
Z_teste = matrizReLU(teste_set_X @ melhor_combinacao[1])
Z_teste = np.c_[np.ones(Z_teste.shape[0]), Z_teste]

O_teste = softPlus(Z_teste @ melhor_combinacao[2])

Erro_teste = (Y_teste - O_teste)
MSE_teste = Erro_teste**2
print(f"Erro teste = {np.sum(Erro_teste)}")



















