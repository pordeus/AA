# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 22:05:52 2021

@author: daniel pordeus
NN
"""

#import random
import numpy as np
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
    sp = np.vectorize(lambda y: np.log(1 + np.exp(y)))
    return sp(x)

def sigmoid(x):
    sig = np.vectorize(lambda y:  (1 - 1 / (1 + np.exp(y))) if y < 0 else  (1 / (1 + np.exp(-y))))
    return sig(x)

def ativTanH(dados):
    matrizSaida = dados.copy()
    for i in range(dados.shape[0]):
        for j in range(dados.shape[1]):
            matrizSaida[i,j] = 1 - np.tanh(dados[i,j])
    return matrizSaida

def dTanh(x):
    dtanh = np.vectorize(lambda y: 1 - np.tanh(y)**2)
    return dtanh(x)

def tanH(x):
    tanh = np.vectorize(lambda y: (np.exp(2*y) -1) / np.exp(2*y) +1)
    return tanh(x)

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


# carregamento dados de entrada
dataset = np.genfromtxt('E:\\Doutorado\\Aulas_Notas e Videos\\AA\\Listas\\Lista 3\\concrete.csv', delimiter=',', skip_header=1)
coluna = dataset.shape[1] - 1

#embaralhando entrada
np.random.shuffle(dataset)

#divisao de conjuntos treinamento, teste e validacao
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
Y_treino = normalizacao01_Y(treino_set[:,coluna])
Y_teste = normalizacao01_Y(teste_set[:,coluna])
Y_valid = normalizacao01_Y(valid_set[:,coluna])

#Hiperparametros
alphas =  [0.001, 0.0001] #np.random.random() # taxa de aprendizagem
hidden = [8, 16, 32]#, 64, 128]
miniBatch = [64, 128, 256]
melhor_combinacao = [1000, [], []] # [erro, Matriz de Pesos W, M]
qtdCombinacoes = len(alphas) * len(hidden) * len(miniBatch)
#
epocas = 300
combinacoes = 0
tamanho_batch = len(treino_norm_X)
MSE = np.zeros((qtdCombinacoes,epocas))
MAE = np.zeros((qtdCombinacoes,epocas))
MRE = np.zeros((qtdCombinacoes,epocas))
MSE_valid = np.zeros(qtdCombinacoes)
print(" ### Inicio de Treinamento ### ")
print(" ### Grid Search ### ")
fig2, ax2 = plt.subplots()
fig2, ax2 = plt.subplots(figsize=(12,8))
for alpha in alphas:
    print(f"Passo de Aprendizagem = {alpha}")
    for h in hidden:
        print(f"Camadas Ocultas = {h}")
        #print(" # # # # # # ")
        # Pesos - aleatorio de 0 a 1 - criação
        W = np.random.rand(dataset.shape[1], h)
        #W = np.zeros((dataset.shape[1], h))
        M = np.random.rand((h+1))
        #M = np.zeros(h+1)
        #MSE = np.zeros(epocas)
        
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
                    #print(f"Epoca={qtdEpocas}")
                    Z = relu(treino @ W)
                
                    Z = np.c_[np.ones(Z.shape[0]), Z]
                
                    O = softPlus(Z @ M) #ativTanH(Z @ M)
                    
                    Erro = (Y_treino_batch - O)/mini
                   
                    #print(f"Erro={np.sum(Erro)}")
                    #Erro_batch += (Erro/ mini)
                    MSE[combinacoes, qtdEpocas] = np.sum((Erro))
                    #MRE[combinacoes, qtdEpocas] = np.sum((Erro) / Y_treino_batch)
                    #print(f"MSE={np.sum(MSE[combinacoes])**2}")
                    #Backprop
                    delta = -Erro #* derivada(Z @ M,"soft_plus")
                    #print(f"delta={delta[0]}")
                    part = delta.reshape(delta.shape[0], 1) @ M[1:].reshape(1, M.shape[0]-1)
                    Chi = derivada(treino @ W,"relu") * part
                    
                    #print(f"Chi={Chi[0]}")
                    M = M - alpha * (delta @ Z)
                    W = W - alpha * treino.T @ Chi
                    qtdEpocas += 1
        #MRE[combinacoes] = MRE[combinacoes] / N
        #MSE = MSE / mini
        MAE[combinacoes] = MSE[combinacoes] / N
        MSE[combinacoes] = MSE[combinacoes]**2 #/ mini
        MSE[combinacoes] = MSE[combinacoes] / (N)
        print(f"Erro = {np.sum(Erro)}  -- MSE = {np.sum(MSE[combinacoes])}")
        #Fim do Treinamento Mini Batch
        
        #desenhar grafico Erro / MSE
        
        # Verificação com Validação
    
        Z_valid = matrizReLU(valid_set_X @ W)
        Z_valid = np.c_[np.ones(Z_valid.shape[0]), Z_valid]
        
        O_valid = softPlus(Z_valid @ M) #ativTanH(Z @ M)
        
        Erro_valid = (Y_valid - O_valid)
        MSE_valid[combinacoes] = np.sum(Erro_valid**2) / len(valid_set_X)
        #MRE_valid = Erro_valid / (Y_valid * len(valid_set_X))
        MAE_valid = Erro_valid / len(valid_set_X)
        print(" ### Resultados da Validação ###")
        print(f"MSE Validação = {MSE_valid[combinacoes]}")
        print(f"RMSE Validação = {np.sqrt(np.sum(MSE_valid))}")
        #print(f"MRE Validação = {np.sum(MRE_valid)}")
        print(f"MAE Validação = {np.sum(MAE_valid)}")
        if np.sum(Erro_valid) < melhor_combinacao[0]:
            melhor_combinacao = [np.sum(Erro_valid), W, M] #armazena o melhor W e M
    
        #Montando gráfico
        ax2.plot(np.arange(epocas), MSE[combinacoes], color=corAleatoria(), label=f"MSE Alpha={alpha} Nh={h}")
        ax2.plot(np.arange(epocas), np.sqrt(MSE[combinacoes]), color=corAleatoria(), label=f"RMSE Alpha={alpha} Nh={h}")
        ax2.plot(np.arange(epocas), MAE[combinacoes], color=corAleatoria(), label=f"MAE Alpha={alpha} Nh={h}")
        ax2.plot(np.arange(epocas), MRE[combinacoes], color=corAleatoria(), label=f"MRE Alpha={alpha} Nh={h}")
        combinacoes += 1
#ax2.set_ylim([0, 5])
ax2.legend()
ax2.set_title('MSE x RMSE x MAE x MRE de Treinamento')
plt.show()

# teste
print(" ### Resultados do Teste ###")
#melhor_combinacao[1] é o melhor W no Grid Search 
Z_teste = matrizReLU(teste_set_X @ melhor_combinacao[1])
Z_teste = np.c_[np.ones(Z_teste.shape[0]), Z_teste]

O_teste = softPlus(Z_teste @ melhor_combinacao[2])

Erro_teste = (Y_teste - O_teste)
MSE_teste = Erro_teste**2 / len(teste_set_X)
#MRE_teste = Erro_teste / (Y_valid * len(teste_set_X))
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
















