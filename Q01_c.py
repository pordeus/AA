# -*- coding: utf-8 -*-
"""
Created on Fri May 21 23:33:38 2021

@author: daniel pordeus
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 20 00:13:11 2021
# Algortimo de Regressão Linear com Gradiente Descendente Estocastico

@author: daniel pordeus

Lista_01 Questão 01 item c
"""

import random
import numpy as np
from matplotlib import pyplot as plt

fileRead = open('E:\\Doutorado\\Aulas_Notas e Videos\\AA\\Listas\\Lista 1\\Q01\\artificial1d.csv', "rt")

table_training = []
for line in fileRead:
    temp = line.split(',')
    temp[0] = float(temp[0])
    temp[1] = float(temp[1])
    table_training.append(temp)

N = len(table_training) #numero de amostras - eixo x
alpha =  0.1 #np.random.random() # taxa de aprendizagem
print("Alpha: " + str(alpha))
wo =  np.random.random()*10 
w1 =  np.random.random()*10

print("Wo Inicial: "+ str(wo))
print("W1 Inicial: "+ str(w1))

X = []
for z in (range(N)):
    X.append(table_training[z][0])

Y = []
for z in range(N):
    Y.append(table_training[z][1])

control = []
for z in range(N):
    control.append(table_training[z][1])

errors = []
for z in (range(N)):
    errors.append(0)

MSE = [(0)]

index = []
for i in range(0, len(table_training)):
    index.append(i)

epoch = 0

#while stable < 5:
while epoch < 150:
    MSE.append(0)
    #print(f"#### EPOCA {epoch} ####")
    random.shuffle(index)
    #print(f"#### Lista de Indices {index} ####")
    #convergence = sum(errors)
    for x in index:
        Y[x] = wo + w1 * X[x]
        errors[x] = table_training[x][1] - Y[x]
        MSE[epoch] = MSE[epoch] + errors[x]**2
        wo = wo + alpha * errors[x]
        w1 = w1 + alpha * errors[x] * X[x]
    MSE[epoch] = MSE[epoch] / (2*N)
    epoch = epoch + 1
    
MSE.pop()

print("Epocas: " + str(epoch))
print("Wo = " + str(wo))
print("w1 = " + str(w1))
print("Erro = " + str(sum(errors)))


def predict(x):
    return wo + w1 * x

predict(0.500)
predict(0.572)
predict(1.613)


Y1 = np.linspace(Y[0], Y[len(Y)-1], len(Y))

fig, ax = plt.subplots()
ax.plot(X, Y1, color='tab:blue')
ax.plot(X, control, 'o', color='tab:red')
ax.set_title('Regressao Linear - GDE')
plt.show()

axis_x = []
for x in range(len(MSE)):
    axis_x.append(x)

fig2, ax2 = plt.subplots()
ax2.plot(axis_x, MSE, color='tab:green')
ax2.set_title('MSE')
plt.show()

fileRead.close()

