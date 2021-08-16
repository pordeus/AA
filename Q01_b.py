# -*- coding: utf-8 -*-
"""
Created on Fri May 21 23:33:38 2021

@author: danie
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 20 00:13:11 2021

@author: daniel pordeus

Lista_01 Questão 01
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
alpha =  0.05 #np.random.random() # taxa de aprendizagem
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

def updateW1(array_error, array_x):
    sum_errors = 0
    for i in range(len(array_error)):
        sum_errors = sum_errors + array_error[i] * array_x[i]
    return sum_errors
        
errors = []
for z in (range(N)):
    errors.append(0)

MSE = []
epoch = 0
convergence = 1
convergence = 1
conv_tax = 0.000001
stable = 0
#while stable < 3:
while epoch < 1200:
    MSE.append(0) ## ?
    #print(f"#### EPOCA {epoch} ####")
    for x in range(0, len(table_training)):
        Y[x] = wo + w1 * X[x]
        errors[x] =  table_training[x][1] - Y[x]
        MSE[epoch] = MSE[epoch] + errors[x]**2
        #print("erro = " + str(errors[x]))
    MSE[epoch] = MSE[epoch] / (2*N)
    wo = wo + alpha * sum(errors) / N
    w1 = w1 + alpha * updateW1(errors, X) / N    
    #convergence = abs(convergence - sum(errors))
    convergence = abs(MSE[epoch] - MSE[epoch-1])
    if convergence < conv_tax:
        stable = stable + 1
    else:
        stable = 0
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

fig, ax = plt.subplots()
ax.plot(X, Y, color='tab:blue', label="Estimado")
ax.plot(X, control, 'o', color='tab:red', label="Original")
ax.set_title('Regressao Linear - GD')
ax.legend()
plt.show()

axis_x = []
for x in range(len(MSE)):
    axis_x.append(x)

fig2, ax2 = plt.subplots()
ax2.plot(axis_x, MSE, color='tab:blue')
ax2.set_title('MSE')
plt.show()

fileRead.close()


