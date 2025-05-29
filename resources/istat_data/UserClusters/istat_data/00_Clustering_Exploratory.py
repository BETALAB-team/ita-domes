import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA

# Import ISTAT 2013 dataset
istat_path = os.path.join(".", "istat_microdata_2013.csv")
data = pd.read_csv(istat_path, delimiter=";")
data.index = data["id"]  # Set index to 'id' column

# Select relevant columns
df = data[['reg',
           'q_1_1_sq1',
           'condpro_1', 'condpro_2',
           'tipolav_1', 'tipolav_2', 
           'livello_1', 'livello_2',
           'titstu_1', 'titstu_2']]

# Preview first 500 rows
df_sample = df.head(500)
n_vars = len(df.columns)

# Fill missing values with column medians
df.fillna(data.median(), inplace=True)

# Normalize data
normalized_values = Normalizer().fit_transform(df.values)

# Pair plot visualization
sns.pairplot(df)
plt.suptitle('Pair Plot of All Columns', y=1.02, fontsize=16)
plt.show()

# Perform PCA
pca = PCA()
pca.fit(normalized_values)
explained_variance = np.round(pca.explained_variance_ratio_ * 100, decimals=1)

# Plot cumulative explained variance
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance) + 1), explained_variance.cumsum(), marker="o", linestyle="--")
plt.grid()
plt.ylabel("Cumulative Percentage of Explained Variance")
plt.xlabel("Number of Components")
plt.title("Explained Variance by Component")
plt.show()

# Reduce number of components
n_components = 5
pca = PCA(n_components=n_components)
pca.fit(normalized_values)
scores_pca = pca.transform(normalized_values)

# Determine optimal number of clusters using WCSS (Within-Cluster Sum of Squares)
WCSS = []
for i in range(1, n_vars):
    kmeans_pca = KMeans(n_clusters=i, init="k-means++", random_state=42)
    kmeans_pca.fit(scores_pca)
    WCSS.append(kmeans_pca.inertia_)

# Plot WCSS to find optimal number of clusters
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_vars), WCSS, marker="o", linestyle="--")
plt.grid()
plt.title("Cluster using PCA Scores")
plt.ylabel("WCSS")
plt.xlabel("Number of Clusters")
plt.show()

# Perform clustering with selected number of components
kmeans_pca = KMeans(n_clusters=n_components, init="k-means++", random_state=42)
kmeans_pca.fit(scores_pca)

# Create DataFrame for PCA scores
df_scores = pd.DataFrame(scores_pca, columns=["comp1", "comp2", "comp3", "comp4", "comp5"])
df_scores['id'] = range(1, 20001)
df_scores.set_index('id', inplace=True)

# Concatenate original dataframe with PCA scores
df_clust_pca_kmeans = pd.concat([df, df_scores], axis=1)

# Assign cluster labels
df_clust_pca_kmeans["segment_kmeans_pca"] = kmeans_pca.labels_
df_clust_pca_kmeans["segment"] = df_clust_pca_kmeans["segment_kmeans_pca"].map({
    0: "Cluster 1", 1: "Cluster 2", 2: "Cluster 3", 3: "Cluster 4", 4: "Cluster 5"
})

# Create subset for visualization
df_visualization = df_clust_pca_kmeans.iloc[:, -7:]
df_visualization['segment'] = df_clust_pca_kmeans['segment']

# Pair plot of PCA components with clustering labels
sns.pairplot(df_visualization, hue='segment')
