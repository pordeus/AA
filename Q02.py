# -*- coding: utf-8 -*-
"""
Created on Fri May 28 17:23:02 2021

@author: daniel pordeus

Lista 01 - Q02
"""

import numpy as np
import matplotlib.pyplot as plt


boston_dataset = np.genfromtxt(
    'E:\\Doutorado\\Aulas_Notas e Videos\\AA\\Listas\\Lista 1\\Q02\\boston.csv',
    delimiter=',', skip_header=1)

boston_hoax = boston_dataset.copy()

np.random.shuffle(boston_hoax)

tamanho_treinamento = int(np.floor((0.8 * len(boston_dataset))))
tamanho_verificacao = len(boston_dataset) - int(np.floor((0.8 * len(boston_dataset))))

set_train = np.zeros((tamanho_treinamento, len(boston_dataset[0])))
set_validation = np.zeros((tamanho_verificacao, len(boston_dataset[0])))

# Letra a
for i in range(0, tamanho_treinamento):
    set_train[i] = boston_hoax[i]

for j in range(0, tamanho_verificacao):
    set_validation[j] = boston_hoax[j+tamanho_treinamento]

def normalizacao01_X(dados):
    dados_norm = np.zeros(dados.shape)
    zero = np.min(dados)
    um = np.max(dados)
    for x in range(dados.shape[0]):
        for y in range(dados.shape[1]):
            dados_norm[x,y] = (dados[x,y] - zero) / (um - zero)
    return dados_norm

def normalizacao01_Y(dados):
    dados_norm = np.zeros(dados.shape)
    zero = np.min(dados)
    um = np.max(dados)
    for x in range(dados.shape[0]):
        dados_norm[x] = (dados[x] - zero) / (um - zero)
    return dados_norm


#normalizando os conjuntos de treinamento e validacao
#set_validation_norm = normalizacao01(set_validation)

y_train = set_train[:,13]
x_train = np.c_[np.ones((set_train.shape[0])), set_train[:, :12]]

y_train_norm = normalizacao01_Y(set_train[:,13])
x_train_norm = normalizacao01_X(set_train[:, :12])
x_train_norm = np.c_[np.ones((x_train_norm.shape[0])), x_train_norm[:, :12]]

#W = np.linalg.solve(x_train_norm.T @ x_train_norm, x_train_norm.T @ y_train_norm)

#pred = x_train @ W

#rmse = np.sqrt(np.mean(((y_train - pred) ** 2)))
#mre = np.mean(np.abs((y_train - pred)/y_train))

#print(f"RMSE = {rmse} e MRE = {mre}")

y_validation = set_validation[:,13]
x_validation = np.c_[np.ones((set_validation.shape[0])), set_validation[:, :12]]

#pred_valid = x_validation @ W

#rmse_valid = np.sqrt(np.mean(((y_validation - pred_valid) ** 2)))
#mre_valid = np.mean(np.abs((y_validation - pred_valid)/y_validation))

#print(f"RMSE = {rmse_valid} e MRE = {mre_valid}")


#GD grau 1
num_epochs = 100
alpha =  0.01

axis_x = []
for x in range(num_epochs):
    axis_x.append(x+1)

new_x_train_norm = x_train_norm
new_x_valid = x_validation
W = np.random.random(new_x_train_norm.shape[1])
epoch = 0
MSE = []
while epoch < num_epochs:
    MSE.append(0)
    Y = new_x_train_norm @ W
    erros = y_train_norm - Y
    MSE[epoch] = (MSE[epoch] + erros**2) / (2*new_x_train_norm.shape[0])
    W = W + alpha * np.sum(erros.reshape(1,new_x_train_norm.shape[0]) @ 
                           new_x_train_norm ) / new_x_train_norm.shape[0]
    epoch = epoch + 1

fig2, ax2 = plt.subplots()
ax2.plot(axis_x, np.sqrt(MSE), color='tab:blue')
ax2.set_title(f"RMSE")
plt.show()
print(f"MSE = {np.sqrt(np.sum(MSE))}")
pred = new_x_valid @ W
fig, ax = plt.subplots()
ax.plot(new_x_valid, pred, color='tab:blue')
ax.plot(new_x_valid, y_validation, 'o', color='tab:red')
ax.set_title(f"Grafico em N={N}")
plt.show()



#GD -N > 1
new_x_train_norm = x_train_norm
new_x_valid = x_validation

Y = np.ones(y_train.shape)
erros = np.ones(y_train.shape)
for N in range(2,12):
    new_x_train_norm = np.concatenate((new_x_train_norm,x_train_norm[:,:12]**N), axis=1)
    new_x_valid = np.concatenate((new_x_valid, new_x_valid[:,:12]**N), axis=1)
    print(f"N={N} forma de x train = {new_x_train_norm.shape}")
    W = np.random.random(new_x_train_norm.shape[1])
    epoch = 0
    MSE = []
    while epoch < num_epochs :
        MSE.append(0)
        Y = new_x_train_norm @ W
        erros = y_train_norm - Y
        MSE[epoch] = (MSE[epoch] + erros**2) / (2*new_x_train_norm.shape[0])
        W = W + alpha * np.sum(erros.reshape(1,new_x_train_norm.shape[0]) @ 
                               new_x_train_norm ) / new_x_train_norm.shape[0]
        epoch = epoch + 1
    fig2, ax2 = plt.subplots()
    ax2.plot(axis_x, np.sqrt(MSE), color='tab:blue', label=f"N={N}")
    ax2.set_title(f"RMSE de N={N}")
    plt.show()
    print(f"MSE = {np.sqrt(np.sum(MSE))}")
    pred = new_x_valid @ W
    fig, ax = plt.subplots()
    ax.plot(new_x_valid, pred, color='tab:blue')
    ax.plot(new_x_valid, y_validation, 'o', color='tab:red')
    ax.set_title(f"Grafico em N={N}")
    plt.show()
    
        
        

