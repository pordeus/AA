# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 14:25:53 2021

@author: daniel pordeus
Lista 4 0 Random Forest
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May  3 23:02:21 2021

@author: danie
"""
#Importando as bibliotecas
import numpy as np
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
#from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from sklearn.metrics import plot_precision_recall_curve

## funcoes úteis
# Função para avaliar o desempenho dos algoritmos. Ele retorna a média das 
# métricas, acurácia no caso.
def avalia_classificador(clf, kf, X, y, f_metrica):
    metrica_val = []
    metrica_train = []
    for train, valid in kf.split(X,y):
        x_train = X[train]
        y_train = y[train]
        x_valid = X[valid]
        y_valid = y[valid]
        clf.fit(x_train, y_train)
        y_pred_val = clf.predict(x_valid)
        y_pred_train = clf.predict(x_train)
        metrica_val.append(f_metrica(y_valid, y_pred_val))
        metrica_train.append(f_metrica(y_train, y_pred_train))
    return np.array(metrica_val).mean(), np.array(metrica_train).mean()

# Para simplificar a apresentação dos resultados e evitar repetição de código criamos a seguinte função auxuliar para imprimir os resultados.
def apresenta_metrica(nome_metrica, metrica_val, metrica_train, percentual = False):
    c = 100.0 if percentual else 1.0
    print('{} (validação): {}{}'.format(nome_metrica, metrica_val * c, '%' if percentual else ''))
    print('{} (treino): {}{}'.format(nome_metrica, metrica_train * c, '%' if percentual else ''))
    
def F1_score(revocacao, precisao):
    return 2*(revocacao*precisao)/(revocacao+precisao)

def novoAvaliaClassificador(y_original, y_previsto):
    falsoPositivo = 0
    verdadeiroPositivo = 0
    falsoNegativo = 0
    verdadeiroNegativo = 0
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
    
    return falsoPositivo, verdadeiroPositivo, falsoNegativo, verdadeiroNegativo

def avalia_classificador_mais_metricas(clf, kf, X, y, f_metrica):
    metrica_val = []
    metrica_train = []
    precisao_val = []
    revocacao_val = []
    precisao_treino = []
    revocacao_treino = []
    
    for train, valid in kf.split(X,y):
        x_train = X[train]
        y_train = y[train]
        x_valid = X[valid]
        y_valid = y[valid]
        clf.fit(x_train, y_train)
        y_pred_val = clf.predict(x_valid)
        y_pred_train = clf.predict(x_train)
        metrica_val.append(f_metrica(y_valid, y_pred_val))
        FP_treino, VP_treino, FN_treino, VN_treino = novoAvaliaClassificador(y_train, y_pred_train)
        FP_val, VP_val, FN_val, VN_val = novoAvaliaClassificador(y_valid, y_pred_val)
        metrica_train.append(f_metrica(y_train, y_pred_train))
        precisao_treino.append(VP_treino / (VP_treino + FP_treino))
        revocacao_treino.append(VP_treino / (VP_treino + FN_treino))
        print(f"Treino Precisao={formataSaida((VP_treino / (VP_treino + FP_treino)))} Revocacao={formataSaida(VP_treino / (VP_treino + FN_treino))}")
        precisao_val.append(VP_val / (VP_val + FP_val))
        revocacao_val.append(VP_val / (VP_val + FN_val))
        print(f"Validação Precisao={formataSaida(VP_val / (VP_val + FP_val))} Revocacao={formataSaida(VP_val / (VP_val + FN_val))}")
        print(f"F1-Score Treino = {F1_score((VP_treino / (VP_treino + FN_treino)), (VP_treino / (VP_treino + FP_treino)))}")
        print(f"F1-Score Validação = {F1_score((VP_val / (VP_val + FN_val)), (VP_val / (VP_val + FP_val)))}")
    return np.array(metrica_val).mean(), np.array(metrica_train).mean(), np.array(precisao_treino).mean(), np.array(revocacao_treino).mean(), np.array(precisao_val).mean(), np.array(revocacao_val).mean()

def rodadaUnica(clf, X, y, f_metrica):
    clf.fit(X, y)
    y_pred_train = clf.predict(X)
    FP_treino, VP_treino, FN_treino, VN_treino = novoAvaliaClassificador(y, y_pred_train)
    metrica_train = (f_metrica(y, y_pred_train))
    precisao_treino = (VP_treino / (VP_treino + FP_treino))
    revocacao_treino = (VP_treino / (VP_treino + FN_treino))
    #print(f"Precisao={formataSaida(precisao_treino)} Revocacao={formataSaida(revocacao_treino)}")
    print(f"F1-Score = {F1_score(precisao_treino, revocacao_treino)}")
    return metrica_train, precisao_treino, revocacao_treino, y_pred_train


def formataSaida(valor):
    saidaFormatada = "{:.2f}".format(valor*100)
    return saidaFormatada + "%"

## MAIN ##
dataset = np.genfromtxt('E:\\Doutorado\\Aulas_Notas e Videos\\AA\\Listas\\Lista 4\\bostonbin.csv', delimiter=',', skip_header=1)
#embaralhando entrada
np.random.shuffle(dataset)

#divisao de conjuntos treinamento, testee e validacao
treino_size = int(np.floor((0.7 * len(dataset))))

treino_set = dataset[0:treino_size,:]
teste_set = dataset[treino_size:,:]

coluna = dataset.shape[1] - 1
#X
X_treino = treino_set[:,:coluna]
X_teste = teste_set[:,:coluna]

#Y
Y_treino = treino_set[:,coluna]
Y_teste = teste_set[:,coluna]

### Random Forest
# Dividindo os dados em 10 folds.
kf = KFold(n_splits=10, shuffle=True)
best = [0, 0] #acuracia, melhor max depth, melhor num estimadores
n_classificadores = np.arange(10, 201, 10)
#fpr = dict()
#tpr = dict()
#i = 0
for nclass in n_classificadores:
    rfc = IsolationForest(n_estimators=nclass)
    print(f"### Estimadores={nclass} ###")

    media_acuracia_val, media_acuracia_train, media_precisao_treino, media_revocacao_treino, media_precisao_val, media_revocacao_val = avalia_classificador_mais_metricas(rfc, kf, X_treino, Y_treino, accuracy_score)
    f1_score_treino = F1_score(media_revocacao_treino, media_precisao_treino)
    f1_score_validacao = F1_score(media_revocacao_val, media_precisao_val)
    apresenta_metrica('F1-Score', f1_score_validacao, f1_score_treino, percentual=False)
    
    #verifico o melhor pelo F1-Score Validacao
    if best[0] <= f1_score_validacao:
        best = [f1_score_validacao, nclass]
    print("#############################")    
print(f"Melhor F1-Score de Validação: {best[0]}, Estimador={best[1]}")

# Rodada completa com melhor gamma e C
print("Rodada completa de Treinamento com melhor Max Depth e Estimadores")
rfc_treino_total = IsolationForest(n_estimators=best[1])
metrica, precisao, revocacao, y_pred_final = rodadaUnica(rfc_treino_total, X_treino, Y_treino, accuracy_score)

print(f"Acurácia: {formataSaida(metrica)}")
print(f"Precisao: {formataSaida(precisao)}")
print(f"Revocacao: {formataSaida(revocacao)}")

# Teste
print("")
print("Treino")
metrica, precisao, revocacao, y_pred = rodadaUnica(rfc_treino_total, X_teste, Y_teste, accuracy_score)
metrics.plot_roc_curve(rfc_treino_total, X_teste, Y_teste)
disp = plot_precision_recall_curve(rfc_treino_total, X_teste, Y_teste)
disp.ax_.set_title('Precision-Recall Binária')

print(f"Acurácia: {formataSaida(metrica)}")
print(f"Precisao: {formataSaida(precisao)}")
print(f"Revocacao: {formataSaida(revocacao)}")

plt.show() 