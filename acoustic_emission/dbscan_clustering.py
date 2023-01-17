import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

# Загрузка данных 

df = pd.read_csv('loc_new3.txt', delimiter = ',')

df1 = df[df.N > 0]
#data = df.drop(['N'], axis = 1)
x = df1.drop(['Y', 'N'], axis = 1).to_numpy()
y = df1.drop(['X', 'N'], axis = 1).to_numpy()

fig, ax = plt.subplots()

# ключ цвета из {'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'}:
ax.scatter(x, y,
           c = 'r',
           s = 1)



ax.set_title('Один цвет')

#  Увеличим размер графика:
fig.set_figwidth(14)
fig.set_figheight(10)


X_principal = df1

db = DBSCAN(eps = 75, min_samples = 40).fit(X_principal)

labels1 = db.labels_



# Создание метки для сопоставления цветов

colours1 = {}

colours1[0] = 'r'

colours1[1] = 'g'

colours1[2] = 'b'

colours1[3] = 'c'

colours1[4] = 'y'

colours1[5] = 'm'

colours1[-1] = 'k'

  

cvec = [colours1[label] for label in labels1]

colors = ['r', 'g', 'b', 'c', 'y', 'm', 'k' ]

  

r = plt.scatter(

        X_principal['X'], X_principal['Y'], marker ='o', color = colors[0])

g = plt.scatter(

        X_principal['X'], X_principal['Y'], marker ='o', color = colors[1])

b = plt.scatter(

        X_principal['X'], X_principal['Y'], marker ='o', color = colors[2])

c = plt.scatter(

        X_principal['X'], X_principal['Y'], marker ='o', color = colors[3])

y = plt.scatter(

        X_principal['X'], X_principal['Y'], marker ='o', color = colors[4])

m = plt.scatter(

        X_principal['X'], X_principal['Y'], marker ='o', color = colors[5])

k = plt.scatter(

        X_principal['X'], X_principal['Y'], marker ='o', color = colors[6])

  

plt.figure(figsize =(14, 10))

plt.scatter(X_principal['X'], X_principal['Y'], c = cvec)

plt.legend((r, g, b, c, y, m, k),

           ('Label 0', 'Label 1', 'Label 2', 'Label 3', 'Label 4', 'Label 5', 'Label -1'),

           scatterpoints = 1,

           loc ='upper left',

           ncol = 3,

           fontsize = 8)

plt.show()
