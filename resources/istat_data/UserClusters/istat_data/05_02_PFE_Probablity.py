import os
import numpy as np
import pandas as pd
import random
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

# Load Data
clusters = pd.read_csv('clusters.csv')
istat_path = os.path.join('.', 'istat_microdata_2013.csv')
data = pd.read_csv(istat_path, delimiter=';')

def process_and_merge_data():
    """Merge and clean datasets for classification analysis."""
    merged_df = pd.merge(data, clusters[['segment', 'normal_l', 'normal_g', 'normal_s', 'identi']], 
                         left_on='id', right_on='identi', how='inner')
    merged_df.drop(columns=['identi'], inplace=True)
    merged_df.reset_index(drop=True, inplace=True)
    return merged_df

data_to_analyze = process_and_merge_data()

# Feature Engineering
data_to_analyze['age'] = data_to_analyze['q_2_4_ric'].map({
    1: [0, 13], 2: [14, 23], 3: [24, 33], 4: [34, 43], 5: [44, 53],
    6: [54, 63], 7: [64, 113], 8: [113, 150], 9: [0, 70]
}).apply(lambda x: random.randint(x[0], x[1]) if isinstance(x, list) else None)

data_to_analyze['area'] = data_to_analyze['q_2_7_class'].map({
    1: [10, 19], 2: [20, 39], 3: [40, 59], 4: [60, 89], 5: [90, 119], 
    6: [120, 149], 7: [150, 400]
}).apply(lambda x: random.randint(x[0], x[1]) if isinstance(x, list) else None)

# Prepare Data for Classification
X = data_to_analyze[['age', 'area']]
y = data_to_analyze[['segment']]
scaler = StandardScaler()
X[['age', 'area']] = scaler.fit_transform(X[['age', 'area']])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Naive Bayes Classifier
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train.values.ravel())
y_pred = nb_classifier.predict(X_test)

# Model Evaluation
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Predict Probabilities
probabilities = nb_classifier.predict_proba(X_test)
probabilities_df = pd.DataFrame(probabilities, columns=[f'Prob_{c}' for c in nb_classifier.classes_])
probabilities_df.to_csv('probabilities_only.csv', index=False)