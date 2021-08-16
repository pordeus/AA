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

def softMax(x):
    sm = np.vectorize(lambda y: (np.exp(y) / np.sum(np.exp(x))))
    return sm(x)

def derivadaTanH(dados):
    matrizSaida = dados.copy()
    for i in range(dados.shape[0]):
        for j in range(dados.shape[1]):
            matrizSaida[i,j] = 1 - np.tanh(dados[i,j])**2
    return matrizSaida

def dTanh(x):
    dtanh = np.vectorize(lambda y: 1 - np.tanh(y)**2)
    return dtanh(x)

def tanH(x):
    tanh = np.vectorize(lambda y: (np.exp(2*y) -1) / np.exp(2*y) +1)
    return tanh(x)

def derivadaSoftMax(x):
    vetorSaida = np.zeros(len(x))
    for i in range(len(x)):
        for j in range(len(x)):
            if i == j:
                vetorSaida[i] = x[i] * (1 - x[i])
            else:
                vetorSaida[i] = -x[i] * x[j]
    return vetorSaida

def derivada(x, function):
        if function == "sigmoid":
            return np.multiply(x, (1-x))
        elif function == "soft_plus":
            return sigmoid(x)
        elif function == "relu":
            d_relu = np.vectorize(lambda y: 0 if y < 0 else 1)
            return d_relu(x)
        elif function == "tanh":
            return dTanh(x)
        elif function == "soft_max":
            return derivadaSoftMax(x)

def crossEntropy(p, q):
    vetorSaida = np.zeros(len(p))
    for j in range(len(p)):
        vetorSaida[j] = -sum([p[i]*np.log(q[i]) for i in range(len(p))])
        #vetorSaida[j] = -np.sum(np.log(p[j])*q[i]) for i in range(len(q[i]))
    return vetorSaida

def hotEncode(x):
    matrizSaida = np.zeros((len(x),len(np.unique(x))))
    for i in range(len(x)):
        matrizSaida[i,int(x[i])] = 1
    return matrizSaida

def cross_entropy(predictions, targets, epsilon=1e-12):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions. 
    Input: predictions (N, k) ndarray
           targets (N, k) ndarray        
    Returns: scalar
    """
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets*np.log(predictions+1e-9))/N
    return ce

def crossEntropyLoss(p, y):
    loss = 0
    for i in range(len(p)):
        for j in range(len(y[i])):
            loss += -y[i,j] * np.log(p[i])
    return loss

def CRLoss(yhat, y):
    saida = np.zeros(len(yhat))
    for i in range(len(yhat)):
        for j in range(len(y[i])):
            if y[i,j] == 1:
                saida[i] += -np.log(yhat[i])
            else:
                saida[i] += -np.log(1 -yhat[i])
    return saida
    

# carregamento dados de entrada
dataset = np.genfromtxt('E:\\Doutorado\\Aulas_Notas e Videos\\AA\\Listas\\Lista 3\\vowel.csv', delimiter=',', skip_header=1)
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
treino_norm_X = normalizacao01_X(treino_set[:,:coluna])
teste_set_X = normalizacao01_X(teste_set[:,:coluna])
valid_set_X = normalizacao01_X(valid_set[:,:coluna])

# Adição da Coluna 1
treino_norm_X = np.c_[np.ones(treino_norm_X.shape[0]), treino_norm_X]
teste_set_X = np.c_[np.ones(teste_set_X.shape[0]), teste_set_X]
valid_set_X = np.c_[np.ones(valid_set_X.shape[0]), valid_set_X]

#Y's
Y_treino = hotEncode(treino_set[:,coluna])
Y_teste = hotEncode(teste_set[:,coluna])
Y_valid = hotEncode(valid_set[:,coluna])

#Hiperparametros
alphas =  [0.1, 0.01]#, 0.0001] #np.random.random() # taxa de aprendizagem
hidden = [8, 16, 32]
miniBatch = [64, 128, 256]
melhor_combinacao = [10000, [], []] # [erro, Matriz de Pesos W, M]
qtdCombinacoes = len(alphas) * len(hidden) * len(miniBatch)
#
k = Y_treino.shape[1] #numero de classes
epocas = 300
combinacoes = 0
print("Inicio de Treinamento")
print("Erros - Grid Search")
tamanho_batch = len(treino_norm_X)
MSE = np.zeros((qtdCombinacoes,epocas))
MAE = np.zeros((qtdCombinacoes,epocas))
MRE = np.zeros((qtdCombinacoes,epocas))
MSE_valid = np.zeros(qtdCombinacoes)
fig2, ax2 = plt.subplots()
fig2, ax2 = plt.subplots(figsize=(12,8))
ax2.set_title('MSE')
for alpha in alphas:
    print("##########################")
    print(f"Passo de Aprendizagem = {alpha}")
    for h in hidden:
        # Pesos - aleatorio de 0 a 1 - criação
        print("--------------------------")
        print(f"Tamanho camada Oculta = {h}")
        print("--------------------------")
        #W = np.random.rand(dataset.shape[1], h)
        W = np.zeros((dataset.shape[1], h))
        #M = np.random.rand((h+1))
        M = np.zeros((h+1,k))
        #MSE = np.zeros(epocas)
        qtdEpocas = 0
        N = len(treino_norm_X)
        for mini in miniBatch:
            Erro = 0
            aux = 0
            #Erro_batch = 0
            #print(f"Mini Batch {mini}")
            while aux < tamanho_batch: #montando intervalos do batch
                inicio = aux
                if (aux + mini) > tamanho_batch:
                    fim = tamanho_batch
                else:
                    fim = aux + mini
                aux += mini
                #print(f"Intervalo {inicio} a {fim}")
                qtdEpocas = 0
                N = len(treino_norm_X)
                treino = treino_norm_X[inicio:fim,:]
                Y_treino_batch = Y_treino[inicio:fim]
                while qtdEpocas < epocas:
                    
                    Z = matrizReLU(treino @ W)
                    
                    Z = np.c_[np.ones(Z.shape[0]), Z]
                    
                    O = softMax(Z @ M)
                    
                    #Erro = cross_entropy(O, Y_treino)
                    Erro = (O * Y_treino_batch) / mini
                    #Erro = Y_treino - (Y_treino * np.log(O))
                    #Erro = Y_treino - O
                    
                    #MSE[qtdEpocas] = np.sum(Erro)
                    MSE[combinacoes, qtdEpocas] = np.sum((Erro))
                    #print(f"MSE[{combinacoes}] = {MSE[combinacoes]}")
                    
                    #Backprop
                    #delta = Erro * derivada(Z @ M,"soft_max")
                    delta = -Erro #np.full(len(Y_treino), -Erro)
                    part = delta @ M[1:].T #.reshape(1, M.shape[0]-1)
                    Chi = derivada(treino @ W,"relu") * part
                    
                    M = M + alpha * (delta.T @ Z).T
                    W = W + alpha * treino.T @ Chi
                    qtdEpocas += 1
        MAE[combinacoes] = MSE[combinacoes] / N
        MSE[combinacoes] = MSE[combinacoes]**2 / N
        print(f"Erro = {np.sum(MSE[combinacoes])}")
        #Fim do Treinamento
        
        # Verificação com Validação
        
        Z_valid = matrizReLU(valid_set_X @ W)
        Z_valid = np.c_[np.ones(Z_valid.shape[0]), Z_valid]
        
        O_valid = softMax(Z_valid @ M) #ativTanH(Z @ M)
        
        Erro_valid =  O_valid * Y_valid #CRLoss(O_valid, Y_valid)
        #Erro_valid = -(Y_valid * np.log(O_valid))
        #Erro_valid = Y_valid - O_valid
        MSE_valid[combinacoes] = np.sum(Erro_valid)**2 / len(valid_set_X)
#        print(f"Erro Validação = {MSE_valid}")
        MAE_valid = Erro_valid / len(valid_set_X)
        print(" ### Resultados da Validação ###")
        print(f"MSE Validação = {MSE_valid[combinacoes]}")
        print(f"RMSE Validação = {np.sqrt(MSE_valid[combinacoes])}")
        #print(f"MRE Validação = {np.sum(MRE_valid)}")
        print(f"MAE Validação = {np.sum(MAE_valid)}")
        if np.sum(Erro_valid) < melhor_combinacao[0]:
            melhor_combinacao = [np.sum(Erro_valid), W, M] #armazena o melhor W
        
        #Montando gráfico
        ax2.plot(np.arange(epocas), MSE[combinacoes], color=corAleatoria(), label=f"MSE Alpha={alpha} Nh={h}")
        ax2.plot(np.arange(epocas), np.sqrt(MSE[combinacoes]), color=corAleatoria(), label=f"RMSE Alpha={alpha} Nh={h}")
        ax2.plot(np.arange(epocas), MAE[combinacoes], color=corAleatoria(), label=f"MAE Alpha={alpha} Nh={h}")
        ax2.plot(np.arange(epocas), MRE[combinacoes], color=corAleatoria(), label=f"MRE Alpha={alpha} Nh={h}")
        combinacoes += 1
#ax2.set_ylim([1.5, 2])
ax2.set_title('MSE x RMSE x MAE x MRE de Treinamento')
ax2.legend()
plt.show()

# teste
#melhor_combinacao[1] é o melhor W no Grid Search 
Z_teste = matrizReLU(teste_set_X @ melhor_combinacao[1])
Z_teste = np.c_[np.ones(Z_teste.shape[0]), Z_teste]

O_teste = softMax(Z_teste @ melhor_combinacao[2])

Erro_teste = O_teste * Y_teste #CRLoss(O_teste, Y_teste)
#Erro_teste  = -(Y_teste * np.log(O_teste))
#Erro_teste = Y_teste - O_teste
MSE_teste = np.sum(Erro_teste)**2 / len(teste_set_X)
MAE_teste = Erro_teste / len(teste_set_X)
print(f"MSE Teste = {np.sum(MSE_teste)}")
print(f"RMSE Teste = {np.sqrt(np.sum(MSE_teste))}")
#print(f"MRE Teste = {np.sum(MRE_teste)}")
print(f"MAE Teste = {np.sum(MAE_teste)}")

#Grafico da Validação
# monta MSE Treinamento
MSE_aux = np.zeros(len(MSE))
indice = 0
for x in MSE:
    MSE_aux[indice] = np.sum(x)
    indice =+ 1

fig, ax = plt.subplots()
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(qtdCombinacoes), np.sqrt(MSE_aux), color=corAleatoria(), label="RMSE Treinamento")
ax.plot(np.arange(qtdCombinacoes), np.sqrt(MSE_valid), color=corAleatoria(), label="RMSE Validação")
ax.legend()
ax.set_title('MSE x RMSE de Validação')
plt.show()

















