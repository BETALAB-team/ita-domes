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

# Load ISTAT dataset and merge
istat_path = os.path.join(".", "istat_microdata_2013.csv")
data = pd.read_csv(istat_path, delimiter=";")
df = pd.merge(data[["id", "rip"]], df, left_on='id', right_on='identi', how='inner')

# Calculate Features
df["loneliness"] = 0.75 * df["max_age"] + 0.25 * df["min_age"] - 10 * df["number"]
df["generation"] = 0.66 * df["max_age"] + 0.33 * df["min_age"] + 10 * df["number"]

# Normalize Data
normal_values = pd.DataFrame()
normal_values["loneliness"] = (df["loneliness"] - df["loneliness"].min()) / (df["loneliness"].max() - df["loneliness"].min())
normal_values["generation"] = (df["generation"] - df["generation"].min()) / (df["generation"].max() - df["generation"].min())
normal_values["studio"] = df["studio_ind"]

# Perform PCA
pca = PCA(n_components=3)
pca.fit(normal_values)
scores_pca = pca.transform(normal_values)

# Clustering
kmeans_pca = KMeans(n_clusters=20, init="k-means++", random_state=42)
kmeans_pca.fit(normal_values)
df["segment"] = kmeans_pca.labels_

# Summary Statistics
summary = df.groupby('segment').agg(
    max_age=('max_age', lambda x: f"{x.mean():.2f}({x.std():.2f})"),
    min_age=('min_age', lambda x: f"{x.mean():.2f}({x.std():.2f})"),
    studio_ind=('studio_ind', lambda x: f"{x.mean():.2f}({x.std():.2f})"),
    number=('number', lambda x: f"{x.mean():.2f}({x.std():.2f})")
).reset_index()
print(summary)

# 3D Scatter Plot
le = LabelEncoder()
segments_encoded = le.fit_transform(df["segment"])
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

# Pair Plots
sns.pairplot(df, hue='segment')
