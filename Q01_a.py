# -*- coding: utf-8 -*-
"""
Created on Fri May 28 16:28:24 2021

@author: daniel pordeus

Lista 01 QUestão 01 letra a
"""

import numpy as np
import matplotlib.pyplot as plt  # Biblioteca para gerar gráficos

artificial_dataset = np.genfromtxt(
    'E:\\Doutorado\\Aulas_Notas e Videos\\AA\\Listas\\Lista 1\\Q01\\artificial1d.csv',
    delimiter=',', skip_header=1)

X = np.c_[np.ones((artificial_dataset.shape[0])), artificial_dataset[:, 0]]
y = artificial_dataset[:, [1]]

#W = np.linalg.inv(X.T @ X) @ X.T @ y
W = np.linalg.solve(X.T @ X, X.T @ y)
W

pred = X @ W

rmse = np.sqrt(np.mean(((y - pred) ** 2)))
mre = np.mean(np.abs((y - pred)/y))

print(f"RMSE = {rmse} e MRE = {mre}")


#plt.hist(np.abs((y-pred)/y))

fig, ax = plt.subplots()
ax.plot(X[:,1], pred, color='tab:blue')
ax.plot(X[:,1], y, 'o', color='tab:red')
ax.set_title('Regressao Linear - OLS')
plt.show()

