# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 15:41:30 2024

@author: vivijac14771
"""

import os
import pandas as pd

from sklearn.cluster import KMeans
import numpy as np

from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import seaborn as sns



#%%

# Import database ISTAT 2013
# istat_path = os.path.join(".", "Resources", "DatiIstat", "istat_microdata_2103.csv")
istat_path = os.path.join(".","istat_microdata_2013.csv") #os.getcwd()
data = pd.read_csv(istat_path, delimiter=";")
data.index = data["id"]  # edifici da 1 a 20000


df = data[['reg',
           'q_1_1_sq1',
           'condpro_1','condpro_2',
           'tipolav_1','tipolav_2', 
           'livello_1','livello_2',
           'titstu_1','titstu_2',
           ]]

#%%
a = df.head(500)
# b = a.to_numpy()
# X = b[:,:10]
# # X = np.array([[1, 2], [1, 4], [1, 0],
#               # [10, 2], [10, 4], [10, 0]])
# kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X)
# kmeans.labels_
# # kmeans.predict([[0, 0], [12, 3]])
# # kmeans.cluster_centers_

n_vars = len(df.columns)

#%% Perform PCA on original dataset


'''Seguo il metodo proposto qua
https://medium.com/@jackiee.jecksom/clustering-and-principal-component-analysis-pca-from-sklearn-c8ea5fed6648

'''

df.fillna(data.median(), inplace = True)
missing = data.isna().sum()

normal_values = Normalizer().fit_transform(df.values)
print(normal_values)


pca = PCA()
pca.fit(normal_values)
per_var = np.round(pca.explained_variance_ratio_*100, decimals = 1)

plt.figure(figsize = (10,6))
plt.plot(range(1, len(per_var)+1), per_var.cumsum(), marker = "o", linestyle = "--")
plt.grid()
plt.ylabel("Percentage Cumulative of Explained Variance")
plt.xlabel("Number of Components")
plt.title("Explained Variance by Component")
plt.show()


#%% Reduce number of components and perform PCA again

n_comp = 5

pca = PCA(n_components = n_comp)
pca.fit(normal_values)

scores_pca = pca.transform(normal_values)

WCSS = []

for i in range(1,n_vars):
  kmeans_pca = KMeans(n_clusters = i, init = "k-means++", random_state = 42)
  kmeans_pca.fit(scores_pca)
  WCSS.append(kmeans_pca.inertia_)
  
plt.figure(figsize = (10,6))
plt.plot(range(1,n_vars), WCSS, marker = "o", linestyle = "--")
plt.grid()
plt.title("Cluster using PCA Scores")
plt.ylabel("WCSS")
plt.xlabel("N Clusters")
plt.show()

#%% Perform clustering with reduced number of components

kmeans_pca = KMeans(n_clusters = n_comp, init = "k-means++", random_state = 42)
kmeans_pca.fit(scores_pca)

#%%
df_scores = pd.DataFrame(scores_pca)
df_scores['id'] = range(1,20001)
df_scores = df_scores.set_index(['id'])

#%% 
# Concatening the original df with the components informations present in scores_pca
df_clust_pca_kmeans = pd.concat([df, df_scores], axis = 1)

# Renaming the column label from each component
df_clust_pca_kmeans.columns.values[-5:] = ["comp1", "comp2", "comp3", "comp4", "comp5"]

# Seting the cluster label to each observation, using the atribute .labels_ 
df_clust_pca_kmeans["segment_kmeans_pca"] = kmeans_pca.labels_

# Mapping each cluster segmentation and renaming their labels 
df_clust_pca_kmeans["segment"] = df_clust_pca_kmeans["segment_kmeans_pca"].map({0:"Cluster 1", 1:"Cluster 2", 2:"Cluster 3", 3:"Cluster 4", 4:"Cluster 5"})

df1 = df_clust_pca_kmeans.iloc[ :, -7:]
df1['segment'] = df_clust_pca_kmeans['segment']
sns.pairplot(df1[0:], hue='segment')

