from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import pdist
import numpy as np

from sklearn.cluster import AgglomerativeClustering


data = pd.read_csv('loc_new3.txt', delimiter = ',')

df1 = data[data.N > 0]

x = df1.drop(['N'], axis = 1).to_numpy()


labelList = range(1, 3061)

 
mergings = linkage(x, method = 'complete')


dendrogram(mergings,
           labels=labelList,
           leaf_rotation=90,
           leaf_font_size=6,
           )

plt.show()

cluster = AgglomerativeClustering(n_clusters = 7, affinity = 'euclidean', linkage = 'ward')
res = cluster.fit_predict(x)
plt.figure(figsize =(14, 10))
plt.scatter(x[:,0],x[:,1], c = cluster.labels_, cmap = 'rainbow')
