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

def build_dict(pred,X):
    tmp = {}
    for i in range(len(pred)):
        c = pred[i]
        if c in tmp:
            tmp[c].append(list(X[i]))
        else:
            tmp[c] = [list(X[i])]
    return tmp


df = pd.read_csv("data_pca.csv") #read the csv of the COAD full raw counts

genes = df["gene"].to_numpy() #keep the gene names in a different list
#print(genes)
genes = genes.tolist() 
#print (genes)

df = df.drop("gene",axis=1)

df = df.T
#df=df.T.reset_index().rename(columns={'index':'var'})
#print((df['var'].to_numpy))
print(df.head())





X = df.to_numpy()

scaler = MinMaxScaler() #normalization
#scaler = Normalizer()
scaler.fit(X)
X= scaler.transform(X)

clustering = KMeans(n_clusters=3)
pred = clustering.fit_predict(X)

tmp = build_dict(pred,X)

plt.figure()
for i in tmp.values():
	i = np.array(i)
	plt.scatter(i[:,0],i[:,1])

plt.scatter(clustering.cluster_centers_[:,0],clustering.cluster_centers_[:,1],s=100,color="red")
plt.show()

labels = clustering.labels_

#df = df.rename(columns={'index':'case'})
df.reset_index(inplace=True)
df = df.rename(columns = {'index':'case'})
print("df.head!!!!!!!!!!!\n", df.head())

#results = pd.DataFrame([df['case'], labels]).T.rename(columns={"0":'case', "1": 'cluster'})
d = {'case':df['case'], 'cluster': labels}
results = pd.DataFrame(data=d)
print(results.head())

df_merge = df.merge(results, how="inner", on="case")
print(df_merge.head())
df_merge.to_csv("merge_kmeans.csv")

#data.gene = pd.Categorical(data.gene)
#data['gene_code'] = data.gene.cat.codes


'''

inertia = []
for i in range(1,11):
	kmeans = KMeans(
		n_clusters=i, init="k-means++",
		n_init=10,
		tol=1e-04, random_state=42
	)
	kmeans.fit(X)
	inertia.append(kmeans.inertia_)
fig = go.Figure(data=go.Scatter(x=np.arange(1,11),y=inertia))
fig.update_layout(title="Inertia vs Cluster Number",xaxis=dict(range=[0,11],title="Cluster Number"),yaxis={'title':'Inertia'},
				annotations=[
			dict(
			x=3,
			y=inertia[2],
			xref="x",
			yref="y",
			text="Elbow!",
			showarrow=True,
			arrowhead=7,
			ax=20,
			ay=-40
		)
	])
'''
