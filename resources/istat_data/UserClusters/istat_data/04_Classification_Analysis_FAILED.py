import os
import numpy as np
import pandas as pd
import ast
import random
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from imblearn.combine import SMOTEENN
import statsmodels.api as sm

# Function Definitions
def random_age_from_range(age_range):
    """Returns a random integer within the given age range."""
    if isinstance(age_range, list) and len(age_range) == 2:
        return random.randint(age_range[0], age_range[1])
    return None

def convert_and_max(array):
    """Returns the maximum value from a list parsed as a numpy array."""
    return np.max(ast.literal_eval(array))

def convert_and_min(array):
    """Returns the minimum value from a list parsed as a numpy array."""
    return np.min(ast.literal_eval(array))

def convert_and_count(array):
    """Returns the count of elements in a list parsed as a numpy array."""
    return np.count_nonzero(ast.literal_eval(array))

# Load Data
clusters = pd.read_csv('clusters.csv')
istat_path = os.path.join('.', 'istat_microdata_2013.csv')
data = pd.read_csv(istat_path, delimiter=';')

def process_and_merge_data():
    """Merge and clean datasets for classification analysis."""
    merged_df = pd.merge(data, clusters[['segment', 'normal_l', 'normal_g', 'normal_s', 'identi']], left_on='id', right_on='identi', how='inner')
    merged_df.drop(columns=['identi'], inplace=True)
    merged_df.reset_index(drop=True, inplace=True)
    return merged_df

data_to_analyze = process_and_merge_data()

# Feature Engineering
data_to_analyze['age'] = data_to_analyze['q_2_4_ric'].map({
    1: [0, 13], 2: [14, 23], 3: [24, 33], 4: [34, 43], 5: [44, 53],
    6: [54, 63], 7: [64, 113], 8: [113, 150], 9: [0, 70]
}).apply(random_age_from_range)

data_to_analyze['area'] = data_to_analyze['q_2_7_class'].map({
    1: [10, 19], 2: [20, 39], 3: [40, 59], 4: [60, 89], 5: [90, 119], 6: [120, 149], 7: [150, 400]
}).apply(random_age_from_range)

# Extreme Case Filtering
extreme_cases = data_to_analyze[(data_to_analyze['age'] > 80) & (data_to_analyze['area'] > 300)]

# Data Preparation for Classification
X = data_to_analyze[['age', 'area']]
y = data_to_analyze[['segment']]
scaler = StandardScaler()
X[['age', 'area']] = scaler.fit_transform(X[['age', 'area']])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model Training and Evaluation
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# SMOTEENN for Imbalanced Data
smote_enn = SMOTEENN()
X_resampled, y_resampled = smote_enn.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)
clf_im = RandomForestClassifier()
clf_im.fit(X_train, y_train)
y_pred_im = clf_im.predict(X_test)
print(confusion_matrix(y_test, y_pred_im))
print(classification_report(y_test, y_pred_im))

# PCA Analysis
pca = PCA(n_components=min(5, X_resampled.shape[1]))
pca.fit(X_resampled)
explained_variance = pca.explained_variance_ratio_
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--', color='b')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.grid(True)
plt.show()

# Extreme Cases PCA Analysis
pca_extreme = PCA(n_components=2)
pca_extreme.fit(extreme_cases[['age', 'area']])
explained_variance_extreme = pca_extreme.explained_variance_ratio_
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(explained_variance_extreme) + 1), explained_variance_extreme, marker='o', linestyle='--', color='r')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio (Extreme Cases)')
plt.grid(True)
plt.show()
