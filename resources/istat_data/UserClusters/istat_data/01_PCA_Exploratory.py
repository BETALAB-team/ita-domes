import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from matplotlib.colors import ListedColormap
import ast
import networkx as nx

# Function Definitions
def convert_and_max(array):
    array = ast.literal_eval(array)
    return np.max(np.array(array))

def convert_and_min(array):
    array = ast.literal_eval(array)
    return np.min(np.array(array))

# Load Data
df = pd.read_csv('family_trees.csv')
df = df[df['number'] != 0]

df["max_age"] = df['age_array'].apply(convert_and_max)
df["min_age"] = df['age_array'].apply(convert_and_min)
df = df.dropna(subset=['studio_ind']).reset_index()

# 3D Scatter Plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['max_age'], df['min_age'], df['number'], c='r', marker='o')
ax.set_title('3D Scatter Plot')
ax.set_xlabel('Max Age')
ax.set_ylabel('Min Age')
ax.set_zlabel('Number')
plt.show()

# Prepare Data for PCA
df = df[["max_age", "min_age", "number", "studio_ind", "identi"]]
df["loneliness"] = 0.75 * df["max_age"] + 0.25 * df["min_age"] - 10 * df["number"]
df["generation"] = 0.66 * df["max_age"] + 0.33 * df["min_age"] + 10 * df["number"]

# Normalize Data
normal_values = pd.DataFrame()
normal_values["loneliness"] = (df["loneliness"] - df["loneliness"].min()) / (df["loneliness"].max() - df["loneliness"].min())
normal_values["generation"] = (df["generation"] - df["generation"].min()) / (df["generation"].max() - df["generation"].min())
normal_values["studio"] = df["studio_ind"]

sns.pairplot(df)
plt.suptitle('Pair Plot of All Columns', y=1.02, fontsize=16)
plt.show()

# Perform PCA
pca = PCA()
pca.fit(normal_values)
explained_variance = np.round(pca.explained_variance_ratio_ * 100, decimals=1)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance) + 1), explained_variance.cumsum(), marker="o", linestyle="--")
plt.grid()
plt.ylabel("Cumulative Percentage of Explained Variance")
plt.xlabel("Number of Components")
plt.title("Explained Variance by Component")
plt.show()

# Reduce number of components and perform PCA again
n_comp = 3
pca = PCA(n_components=n_comp)
pca.fit(normal_values)
scores_pca = pca.transform(normal_values)

# Determine optimal number of clusters using WCSS
WCSS = []
for i in range(1, 10):
    kmeans_pca = KMeans(n_clusters=i, init="k-means++", random_state=42)
    kmeans_pca.fit(scores_pca)
    WCSS.append(kmeans_pca.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 10), WCSS, marker="o", linestyle="--")
plt.grid()
plt.title("Cluster using PCA Scores")
plt.ylabel("WCSS")
plt.xlabel("Number of Clusters")
plt.show()

# Perform clustering
kmeans_pca = KMeans(n_clusters=4, init="k-means++", random_state=42)
kmeans_pca.fit(normal_values)
silhouette_avg = silhouette_score(normal_values, kmeans_pca.labels_)

# Create DataFrame for PCA scores
df_scores = pd.DataFrame(scores_pca, columns=["comp1", "comp2", "comp3"])
df_scores['id'] = range(1, 17856)
df_scores.set_index('id', inplace=True)

df_clust_pca_kmeans = df_scores

df_clust_pca_kmeans["segment_kmeans_pca"] = kmeans_pca.labels_
df_clust_pca_kmeans["segment"] = df_clust_pca_kmeans["segment_kmeans_pca"].astype("str")
df["segment"] = df_clust_pca_kmeans["segment"]

# Pair Plot with Clusters
sns.pairplot(df_clust_pca_kmeans, hue='segment')

# 3D Scatter Plot with Clusters
le = LabelEncoder()
segments_encoded = le.fit_transform(df_clust_pca_kmeans["segment"])
cmap = ListedColormap(plt.cm.get_cmap('viridis', len(le.classes_)).colors)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(df["min_age"], df["max_age"], df["number"], c=segments_encoded, cmap=cmap, marker='o')
legend1 = ax.legend(*scatter.legend_elements(), title="Segments")
ax.add_artist(legend1)
ax.set_title('3D Scatter Plot')
ax.set_xlabel('Min Age')
ax.set_ylabel('Max Age')
ax.set_zlabel('Number')
plt.show()

# Normalize Additional Features
df["normal_l"] = normal_values["loneliness"]
df["normal_g"] = normal_values["generation"]
df["normal_s"] = normal_values["studio"]
df["new1"] = df["min_age"] + 0.25 * (df["max_age"] - df["min_age"]) - 10 * df["number"]
df["new2"] = 100 * df["studio_ind"]

sns.pairplot(df)
sns.pairplot(df[["max_age", "min_age", "number", "studio_ind", "segment"]], hue="segment")
