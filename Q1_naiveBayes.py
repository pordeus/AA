# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 09:01:59 2021

@author: Daniel Pordeus
Fonte de Consulta: 
    https://medium.com/computando-arte/naive-bayes-teoria-e-implementa%C3%A7%C3%A3o-do-zero-e302976538af
Lista 2 - Naive Bayes
"""


import numpy as np
from sklearn.model_selection import KFold
#from sklearn.model_selection import train_test_split
#from sklearn.datasets import load_breast_cancer

breastcancer_dataset = np.genfromtxt(
    'E:\\Doutorado\\Aulas_Notas e Videos\\AA\\Listas\\Lista 2\\breastcancer.csv',
    delimiter=',', skip_header=1)

breastcancer = breastcancer_dataset.copy()

### Criando uma função para extrair os indices das observações de cada classe no conjunto de dados. 
def indices_data(y): ## Recebe os targets 
    n_classes = len(np.unique(y))
    classes_indices = []
    for i in range(n_classes): ## As classes devem ser ordenadas a partir de 0.
        classes_indices.append(np.where(y==i)[0].tolist()) ## Faz um append dos indices da classe i
    return classes_indices

def prob_classes(y,classes_indices):
    probs_class = []
    for i in range(len(classes_indices)):
        probs_class.append(len(classes_indices[i])/len(y))
    return probs_class


### Estimativa da média e variância para cada variável, condicionalmente a cada classe.
def estimativas(X,classes_indices):
    n_classes = len(classes_indices) ## Número de classes
    n_variaveis = X.shape[1]  ## quantidade de variáveis do conjunto
    estimativas_medias = np.zeros((n_classes,n_variaveis),dtype=np.float64) ## Para salvar as estimativas de médias p/ cada variável condicional a cada classe
    estimativas_var = np.zeros((n_classes,n_variaveis),dtype=np.float64) ## Para salvar as estimativas de variância p/ cada variável condicional a cada classe
    for i in range(n_classes):
        estimativas_medias[i,:] = X[classes_indices[i]].mean(axis=0) ## Calcula a média de cada variável condicional a classe i 
        estimativas_var[i,:] = X[classes_indices[i]].var(axis=0) ## Calcula a var de cada variável condicional a classe i 
    return estimativas_medias,estimativas_var


def gaussiana(media,var,x):
    return (1/np.sqrt(2*(np.pi)*var))*np.exp(-((x-media)**2)/(2*var))

def ajusta_naivebayes(X_treino,y_treino):
    #n_classes = len(np.unique(y_treino)) ## Número de classes 
    classes_indices = indices_data(y_treino) ## Indices nos dados de cada classe
    probs_class = prob_classes(y_treino,classes_indices) ## Probs marginais estimativas
    media, var = estimativas(X_treino,classes_indices) ## Media e variancia estimadas no conjunto de treino
    return media,var,probs_class

def prediz_naivebayes(X_teste,media,var,probs_class):
    #### Usando as estimativas do conjunto treino para predizer a classe das observações do teste
    n_classes = media.shape[0]
    n_observacoes = X_teste.shape[0]
    probabilidade_predita = np.zeros((n_observacoes,n_classes),dtype=np.float64) ### Probabilidade predita das observações serem de cada uma das classes
    for i in range(n_classes):
        probabilidade_predita[:,i] = np.sum(np.log(gaussiana(media[i,:],var[i,:],X_teste)),axis=1) + np.log(probs_class[i]) 
    return np.argmax(probabilidade_predita,axis=1)

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
i = 0
acuraciaFinal = 0
kf = KFold(n_splits=K, shuffle=True, random_state=5)
for train, valid in kf.split(X,y):
    x_treinamento_norm = X[train]#normalizacao01_X(X[train])
    y_treinamento = y[train]
    x_teste_norm = X[valid]#normalizacao01_X(X[valid])
    y_teste = y[valid]
    y_estimado = np.zeros(y_teste.shape)
    media,var, prob_classes_n = ajusta_naivebayes(x_treinamento_norm, y_treinamento)
    y_pred = prediz_naivebayes(x_teste_norm, media, var, prob_classes_n)
    #classe_estimada_x = knn(k,x,x_treinamento_norm, y_treinamento)
    #acuracia = acuracia + np.mean(y_pred == y_teste)
    acuracia, falsoPositivo, verdadeiroPositivo, falsoNegativo, verdadeiroNegativo = avaliaClassificador(y_teste, np.round(y_pred))
    print(f"A acurácia no FOLD {i+1} foi de {formataSaida(acuracia)}")
    #print(f"FP={falsoPositivo}, VP={verdadeiroPositivo}, FN={falsoNegativo}, VN={verdadeiroNegativo}")
    precisao = verdadeiroPositivo / (verdadeiroPositivo + falsoPositivo)
    format_precisao = "{:.2f}".format(precisao)
    revocacao = verdadeiroPositivo / (verdadeiroPositivo + falsoNegativo)
    format_revocacao = "{:.2f}".format(revocacao)
    print(f"Precisao={formataSaida(precisao)} Revocacao={formataSaida(revocacao)}")
    acuraciaFinal = acuraciaFinal + acuracia
    print(f"F1-Score={F1_score(revocacao, precisao)}")
    i = i + 1
acuraciaFinal = acuraciaFinal / K
print(f"A acurácia Média dos K-Fold =  {formataSaida(acuraciaFinal)}")









