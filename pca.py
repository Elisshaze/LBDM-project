from numpy.lib.utils import source
import pandas as pd
from pandas import *
import plotly.graph_objects as go
import matplotlib.pyplot as plt 
import re
import numpy as np
import itertools
import sys, getopt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.decomposition import PCA

def build_dict(pred,X):
    tmp = {}
    for i in range(len(pred)):
        c = pred[i]
        if c in tmp:
            tmp[c].append(list(X[i]))
        else:
            tmp[c] = [list(X[i])]
    return tmp


df_cases = pd.read_csv("merge_kmeans.csv")
print(df_cases.head())

X = df_cases.iloc[:, 2:19949].values
y= df_cases.iloc[:, 19949].values
print (X, y)

pca = PCA()

# fit and transform data
X_pca = pca.fit_transform(X)
'''
plt.figure()
percent_variance = np.round(pca.explained_variance_ratio_* 100, decimals =2)
plt.bar(np.arange(len(percent_variance)), height=percent_variance)
plt.ylabel("variance")
plt.xlabel("Principal Component")
plt.yscale("log")
plt.show()
'''
pca = PCA(n_components=2)

# fit and transform data
X_pca = pca.fit_transform(X)
tmp = build_dict(y,X_pca)
c = 1
for i in tmp.values():
	i = np.array(i)
	plt.scatter(i[:,0],i[:,1],label="type "+str(c))
	c = c + 1
plt.legend()
plt.show()