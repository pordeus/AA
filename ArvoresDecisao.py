# -*- coding: utf-8 -*-
"""
Created on Mon May  3 23:01:40 2021
aluno: Daniel Pordeus
@author: Curso de AA Petrobras 2019
Lista 02 - Árvores de Decisão
"""

### Árvore de decisão
#https://scikit-learn.org/0.17/modules/generated/sklearn.tree.DecisionTreeClassifier.html
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn import tree
import numpy as np
import graphviz 

# Para simplificar a apresentação dos resultados e evitar repetição de código criamos a seguinte função auxuliar para imprimir os resultados.
def apresenta_metrica(nome_metrica, metrica_val, metrica_train, percentual = False):
    c = 100.0 if percentual else 1.0
    print('{} (treino): {}{}'.format(nome_metrica, metrica_train * c, '%' if percentual else ''))
    print('{} (validação): {}{}'.format(nome_metrica, metrica_val * c, '%' if percentual else ''))

def F1_score(revocacao, precisao):
    return 2*(revocacao*precisao)/(revocacao+precisao)

def avaliaClassificador(y_original, y_previsto):
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

def formataSaida(valor):
    saidaFormatada = "{:.2f}".format(valor*100)
    return saidaFormatada + "%"

# Assim como no laboratório anterior, aqui temos uma função para avaliar o desempenho dos algoritmos sem termos de ficar repetindo código. Ele retorna a média das métricas, acurácia no caso.
#Função idêntica à usada nos modelos de regressão.
def avalia_classificador(clf, kf, X, y, f_metrica):
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
        FP_treino, VP_treino, FN_treino, VN_treino = avaliaClassificador(y_train, y_pred_train)
        FP_val, VP_val, FN_val, VN_val = avaliaClassificador(y_valid, y_pred_val)
        metrica_train.append(f_metrica(y_train, y_pred_train))
        precisao_treino.append(VP_treino / (VP_treino + FP_treino))
        revocacao_treino.append(VP_treino / (VP_treino + FN_treino))
        print(f"Precisao={formataSaida((VP_treino / (VP_treino + FP_treino)))} Revocacao={formataSaida(VP_treino / (VP_treino + FN_treino))}")
        precisao_val.append(VP_val / (VP_val + FP_val))
        revocacao_val.append(VP_val / (VP_val + FN_val))
        print(f"Precisao={formataSaida(VP_val / (VP_val + FP_val))} Revocacao={formataSaida(VP_val / (VP_val + FN_val))}")
        print(f"F1-Score Treino = {F1_score((VP_treino / (VP_treino + FN_treino)), (VP_treino / (VP_treino + FP_treino)))}")
        print(f"F1-Score Validação = {F1_score((VP_val / (VP_val + FN_val)), (VP_val / (VP_val + FP_val)))}")
    return np.array(metrica_val).mean(), np.array(metrica_train).mean(), np.array(precisao_treino).mean(), np.array(revocacao_treino).mean(), np.array(precisao_val).mean(), np.array(revocacao_val).mean()
    

breastcancer = np.genfromtxt(
    'E:\\Doutorado\\Aulas_Notas e Videos\\AA\\Listas\\Lista 2\\breastcancer.csv',
    delimiter=',', skip_header=1)
#breastcancer.head(3)
#breastcancer = breastcancer.sort_values('Survived')

# aqui montamos a matriz de atributos X e o vetor coluna de respostas Y.
# Note que não selecionamos algumas colnas, como Nome e Ticket
y = breastcancer[:,30]
X = breastcancer[:, :29]

kf = KFold(n_splits=10, shuffle=True, random_state=5)

dt = tree.DecisionTreeClassifier(max_depth=3)

media_acuracia_val, media_acuracia_train, media_precisao_treino, media_revocacao_treino, media_precisao_val, media_revocacao_val = avalia_classificador(dt, kf, X, y, accuracy_score) 
apresenta_metrica('Acurácia', media_acuracia_val, media_acuracia_train, percentual=True)
apresenta_metrica('Precisão', media_precisao_val, media_precisao_treino, percentual=True)
apresenta_metrica('Revogacão', media_revocacao_val, media_revocacao_treino, percentual=True)
f1_score_treino = F1_score(media_revocacao_treino, media_precisao_treino)
f1_score_validacao = F1_score(media_revocacao_val, media_precisao_val)
apresenta_metrica('F1-Score', f1_score_validacao, f1_score_treino, percentual=False)

#media_auc_val, media_auc_train = avalia_classificador(dt, kf, X, y, roc_auc_score) 
#apresenta_metrica('AUC', media_auc_val, media_auc_train, percentual=True)

dot_data = tree.export_graphviz(dt, out_file=None, 
                                feature_names=['1', '2', '3', '4', '5','6','7',
                                               '8','9', '10', '11', '12', '13',
                                               '14','15','16', '17', '18', '19', 
                                               '20','21', '22', '23', '24', '25',
                                               '26','27', '28', '29'],  
                                filled=True, rounded=True,  
                                special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 
