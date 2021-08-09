# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 20:17:57 2021

@author: daniel pordeus
Lists 5 - Q1 - k-médias

Instrução:
Para escolher entre Euclidiana ou Mahalanobis, basta alterar a linha 57
"""

import numpy as np
#import heapq
import random
import matplotlib.pyplot as plt
from sklearn.metrics import davies_bouldin_score
from scipy.spatial.distance import cdist

def corAleatoria():
    red = random.random()
    green = random.random()
    blue = random.random()
    return (red, green, blue)

def normalizacao(dados):
    dados_norm = np.zeros(dados.shape)
    zero = np.min(dados)
    um = np.max(dados)
    for x in range(dados.shape[0]):
        dados_norm[x] = (dados[x] - zero) / (um - zero)
    return dados_norm

def distEuclidiana(x, X):
    distancia = []
    for i in range(len(X)):
        distancia.append(np.sqrt((x[0] - X[i,0])**2 + (x[1] - X[i,1])**2))
    return np.array(distancia)

def distMahalanobis(x,y, X):
    a = x.reshape(x.shape[0],1)
    b = y.reshape(y.shape[0],1)
    ab = np.concatenate((a,b), axis=1)
    return cdist(ab, X, 'mahalanobis')

dataset = np.genfromtxt(
    'E:\\Doutorado\\Aulas_Notas e Videos\\AA\\Listas\\Lista 5\\quake.csv',
    delimiter=',', skip_header=1)

np.random.shuffle(dataset)

#X = normalizacao(dataset[:, 0])
#Y = normalizacao(dataset[:, 1])

Agrupamentos = np.arange(4,21)
agrupamento = 0
repeticoes = 50
metrica = 0 # 0 = Mahalanobis| 1 = Euclidiana 
indice_DB = np.zeros(len(Agrupamentos))
classes = np.zeros((len(Agrupamentos), repeticoes, len(dataset)))
for K_classes in Agrupamentos:
    #centroides aleatorios
    #K_x = np.random.rand(K_classes)
    #K_y = np.random.rand(K_classes)
    print(f"Qtd Classes: {K_classes}")
    #print(f"K_x iniciais = {K_x}")
    #print(f"K_y iniciais = {K_y}")
    K_x = np.zeros(K_classes)
    K_y = np.zeros(K_classes) 
    
    #tratamento da entrada
    dists = np.zeros((K_classes, len(dataset)))
    normalizado = dataset.copy()
    normalizado[:,0] = normalizacao(normalizado[:,0])
    normalizado[:,1] = normalizacao(normalizado[:,1])
    
    #Centroides iniciam com os primeiros k valores
    for i in range(K_classes):
        K_x[i] = normalizado[i, 0]
        K_y[i] = normalizado[i ,1]
    
    # INICIO
    rep = 0
    while rep < repeticoes:
        #Matriz de distancias para cada centroide
        k = 0
        if metrica == 0: # Mahalanobis
            dists = distMahalanobis(K_x, K_y, normalizado)
        else: # Euclidiana
            while k < K_classes:
                dists[k] = distEuclidiana((K_x[k], K_y[k]), normalizado)
                k += 1
        
        #Seleciona a menor distancia para cada ponto 
        #Identifica o ponto pelo indice
        for i in range(len(dataset)):
            classes[agrupamento, rep, i] = int(np.argmin(dists[:,i]))
            
        indice_DB[agrupamento] += davies_bouldin_score(normalizado, classes[agrupamento,rep])

        k = 0
        while k < K_classes:
            tamanho = np.count_nonzero(classes[agrupamento, rep] == k)
            #print(f"Tamanho da classe {k} = {tamanho}")
            cont_tamanho = 0
            aux = np.zeros((tamanho, 2))
            for i in range(len(classes[agrupamento, rep])):
                if classes[agrupamento, rep, i] == k:
                    aux[cont_tamanho] = (normalizado[i,0], normalizado[i,1])
                    #print(f"Aux[{cont_tamanho}] = ({normalizado[i,0], normalizado[i,1]})")
                    cont_tamanho += 1
            #print(f"Aux[] = {aux}")
            K_x[k] = np.mean(aux[:,0])
            K_y[k] = np.mean(aux[:,1])
            #print(f"K_x[{k}] = {K_x[k]}")
            #print(f"K_y[{k}] = {K_y[k]}")
            k += 1

        rep += 1
    indice_DB[agrupamento] = indice_DB[agrupamento] / repeticoes
    print(f"Indice DB = {indice_DB[agrupamento]}")
    agrupamento += 1
    

#Agrupamento com melhor indice DB
melhor = np.argmin(indice_DB)
K_melhor = Agrupamentos[melhor]

#Monta a matriz de cores
cores = []
for cor in range(K_melhor):
    cores.append(corAleatoria())

#Obtem conjunto de rotulacao com menor indice DB dentre o melhor Agrupamento
menor_DB = 10
melhor_conjunto = np.zeros(len(dataset))
for x in classes[melhor]:
    i_DB = davies_bouldin_score(normalizado, x)
    if menor_DB > i_DB:
        melhor_conjunto = x
        menor_DB = i_DB

fig2, ax2 = plt.subplots(figsize=(12,8))
fig2, ax2 = plt.subplots()
#monta os grupos de classes para plotagem
Centroides_x = np.zeros(K_melhor)
Centroides_y = np.zeros(K_melhor) 

for i in range(K_melhor):
    Centroides_x[i] = normalizado[i, 0]
    Centroides_y[i] = normalizado[i ,1]

k_m = 0
while k_m < K_melhor:
    tamanho = np.count_nonzero(melhor_conjunto == k_m)
    cont_tamanho = 0
    aux = np.zeros((tamanho, 2)) # X, Y
    for i in range(len(melhor_conjunto)):
        if melhor_conjunto[i] == k_m:
            aux[cont_tamanho] = (normalizado[i,0], normalizado[i,1])
            cont_tamanho += 1
    Centroides_x[k_m] = np.mean(aux[:,0])
    Centroides_y[k_m] = np.mean(aux[:,1])
    ax2.plot(aux[:,0], aux[:,1], 'o', color=cores[k_m])#, label=f"Classe {k_m}")
    k_m += 1

ax2.plot(Centroides_x, Centroides_y, '*', color="black", label="Centroides")
ax2.set_title('Latitudes x Longitudes')
ax2.legend()
plt.show()

#Plot do Indice DB
fig, ax = plt.subplots(figsize=(12,8))
ax.set_title('Indice DB x Tam. Agrupamento')
ax.plot(Agrupamentos, indice_DB, color="black")
plt.show()

