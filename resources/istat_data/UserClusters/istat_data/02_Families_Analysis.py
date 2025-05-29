import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.decomposition import PCA
import networkx as nx

# Function Definitions
def replace_age_values(array):
    if not isinstance(array, np.ndarray):
        raise ValueError("Input must be a numpy array.")
    result = array.copy()
    indices = (result == 0)
    result[indices] = np.random.randint(11, 15, size=np.sum(indices))
    for i in range(10):
        indices = (result == 15 + i * 10)
        result[indices] = np.random.randint(20 + i * 10, 25 + i * 10, size=np.sum(indices))
    return result.tolist()

def replace_stud_values(array):
    if not isinstance(array, np.ndarray):
        raise ValueError("Input must be a numpy array.")
    result = array.copy()
    mapping = {1: 0, 3: 5, 4: 8, 5: 12, 7: 16, 9: 18, 10: 21, 99: 99}
    for key, value in mapping.items():
        result[result == key] = value
    return result

def replace_exstud_values(array):
    if not isinstance(array, np.ndarray):
        raise ValueError("Input must be a numpy array.")
    result = array.copy()
    result[array < 10] = 0
    result[(array >= 10) & (array < 15)] = 5
    result[(array >= 15) & (array < 19)] = 8
    result[(array >= 19) & (array < 23)] = 12
    result[(array >= 23) & (array < 25)] = 16
    result[(array >= 25) & (array < 29)] = 18
    result[array >= 29] = 21
    return result

warnings.filterwarnings("ignore")

# Load Data
istat_path = os.path.join(".", "istat_microdata_2013.csv")
data = pd.read_csv(istat_path, delimiter=";")
data.index = data["id"]

df_number = data.iloc[:, 89].to_frame()
df_number.rename(columns={'q_1_1_sq1': 'number'}, inplace=True)
df_number["identi"] = data.index
df_number['age_array'] = None
df_number['studio'] = None
df_number['exstudio'] = None
df_number['studio_ind'] = None
df_number['relation_matrix'] = None

# Processing Data
for i in range(len(df_number)):
    relations = data.iloc[i, 25:37].to_list()
    df_number.at[i, 'age_array'] = replace_age_values(data.iloc[i, 13:25].to_numpy())
    df_number.at[i, 'studio'] = replace_stud_values(data.iloc[i, 73:85].to_numpy())
    df_number.at[i, 'exstudio'] = replace_exstud_values(data.iloc[i, 13:25].to_numpy())
    df_number.at[i, 'exstudio'] = df_number.at[i, 'studio'] / df_number.at[i, 'exstudio']
    df_number.at[i, 'studio_ind'] = np.mean(df_number.at[i, 'exstudio'])

# Flatten and Process Matrices for PCA
df_number['matrix'] = None
n_age, n_stud = 10, 7
for i in range(len(df_number)):
    df_number.at[i, 'matrix'] = np.zeros((n_age+1, n_stud+1), dtype=int)
    for j in range(int(df_number.at[i, "number"])):
        could_study = replace_exstud_values(data.iloc[i, 13:25].to_numpy())
        first_indice = df_number.at[i, "age_array"][j] // (100 // n_age)
        year_stud = df_number.at[i, "studio"][j]
        g = min((0 if could_study[j] == 0 else year_stud / could_study[j]), 1)
        second_indice = int((g * 100) // (100/n_stud))
        df_number.at[i, 'matrix'][first_indice, second_indice] += 1

# Flattening for PCA
df_number['flattened'] = df_number['matrix'].apply(lambda x: x.flatten())
matrix_data = np.vstack(df_number['flattened'].values)

# Apply PCA
pca = PCA(n_components=5)
pca.fit(matrix_data)
explained_variance = pca.explained_variance_ratio_

# Explained Variance Plot
plt.figure(figsize=(8, 6))
plt.plot(np.arange(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--', color='b')
plt.title('Explained Variance by Principal Components')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.grid(True)
plt.show()

# Cumulative Explained Variance Plot
cumulative_variance = np.cumsum(explained_variance)
plt.figure(figsize=(8, 6))
plt.plot(np.arange(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='-', color='r')
plt.title('Cumulative Explained Variance by Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.show()
