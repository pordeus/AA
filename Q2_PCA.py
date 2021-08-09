# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 20:18:28 2021

@author: daniel pordeus
Lista 5 - Q2 - PCA

Seguindo passo a passo do site:
https://builtin.com/data-science/step-step-explanation-principal-component-analysis    
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Funcoes Uteis
def normalizacao(dados):
    dados_norm = np.zeros(dados.shape)
    media = np.mean(dados)
    desvio = np.std(dados)
    for x in range(dados.shape[0]):
        dados_norm[x] = (dados[x] - media) / (desvio)
    return dados_norm

def formataSaida(valor):
    saidaFormatada = "{:.2f}".format(valor*100)
    return saidaFormatada + "%"

# Inicio
pinguins = np.genfromtxt(
    'E:\\Doutorado\\Aulas_Notas e Videos\\AA\\Listas\\Lista 5\\penguins.csv',
    delimiter=',', skip_header=1)

np.random.shuffle(pinguins)

pandas_dataset = pd.DataFrame(pinguins, columns = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'classe'])

print(pandas_dataset.head())

# Etapa 1 - Normalizacao
comprimento_bico = normalizacao(pinguins[:, 0]).reshape(pinguins.shape[0], 1)
largura_bico = normalizacao(pinguins[:, 1]).reshape(pinguins.shape[0], 1)
comprimento_flipper = normalizacao(pinguins[:, 2]).reshape(pinguins.shape[0], 1)
massa_corporea = normalizacao(pinguins[:, 3]).reshape(pinguins.shape[0], 1)
classe = pinguins[:,4].reshape(pinguins.shape[0], 1)

novo_conjunto = np.concatenate((comprimento_bico, largura_bico, comprimento_flipper, massa_corporea, classe), axis=1)

novo_conjunto_norm = pd.DataFrame(novo_conjunto, columns = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'classe'])

print(novo_conjunto_norm.head())

#Etapa 2 - Matriz de Covariancia

conjunto_variaveis = np.concatenate((comprimento_bico, largura_bico, comprimento_flipper, massa_corporea), axis=1)

covariancia_inicial = np.cov(conjunto_variaveis.T)

# Etapa 3 - Autovetores e autovalores

auto_decomposicao = np.linalg.eig(covariancia_inicial)
auto_valores = auto_decomposicao[0] 
auto_vetores = auto_decomposicao[1]

# Etapa 4 - Vetor de Características
PCs = auto_valores / np.sum(auto_valores)
#Apresenta o ranking dos Componentes Principais (PCs)
print("Principal Components")
i = 1
for x in PCs:
    print(f" PC{i} {formataSaida(x)}")
    i += 1

PC1 = auto_vetores[np.argmax(auto_valores)]
PC2 = auto_vetores[np.argmax(auto_valores[auto_valores<np.max(auto_valores)])+1]
valor_3 = np.sort(auto_valores)[::-1][2] #pega o valor do 3o maior
posicao_valor_3 = np.argwhere(auto_valores == valor_3)[0][0]
PC3 = auto_vetores[posicao_valor_3]

# Etapa 5 - Novo dataset

#Dimensão 1
penguins_final = conjunto_variaveis @ PC1

#print(np.cov(penguins_final))
print(f"Variancia Explicada D1: {auto_valores[np.argmax(auto_valores)]}")
#Dimensão 2
D2 = np.concatenate((PC1.reshape(PC1.shape[0], 1), PC2.reshape(PC2.shape[0], 1),), axis=1)
penguins_final_D2 = conjunto_variaveis @ D2
#print(np.cov(penguins_final_D2.T))
Var2 = auto_valores[np.argmax(auto_valores)] + auto_valores[np.argmax(auto_valores[auto_valores<np.max(auto_valores)])+1]
print(f"Variancia Explicada D2: {Var2}")
# Dimensão 3
D3 = np.concatenate((D2, PC3.reshape(PC3.shape[0], 1),), axis=1)
penguins_final_D3 = conjunto_variaveis @ D3
#print(np.cov(penguins_final_D2.T))
Var3 = valor_3 + Var2
print(f"Variancia Explicada D2: {Var3}")

#Dimensão 4 - Original
print(f"Variancia Explicada D4: {np.sum(auto_valores)}")

# Gráficos
X = pinguins[:, np.argmax(auto_valores)]
Y = classe.reshape(classe.shape[0])
# Grafico com coluna principal selecionada
plt.scatter(X,Y)
# Grafico com PC1
plt.scatter(penguins_final,Y) # Letra A

fig, ax = plt.subplots(figsize=(12,8))
ax.set_title('PCA Dimensão 1')
ax.plot(X, Y, 'o', color="red", label="Valores Originais")
ax.plot(penguins_final, normalizacao, 'x', color="blue", label="PC1")
ax.legend()
plt.show()
