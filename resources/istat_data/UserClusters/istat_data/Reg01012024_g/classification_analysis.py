# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 14:22:24 2024

@author: khajmoh18975
"""


import networkx as nx
import matplotlib.pyplot as plt
import os
import pandas as pd
import ast
import numpy as np
import random
import seaborn as sns
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


#%%
def random_age_from_range(age_range):
    if isinstance(age_range, list) and len(age_range) == 2:
        return random.randint(age_range[0], age_range[1])
    return None

def convert_and_max(array):
    # Convert each element in the array to a float
    array=ast.literal_eval(array)
    numeric_array = np.array(array)
    return np.max(numeric_array)
def convert_and_count(array):
    # Convert each element in the array to a float
    array=ast.literal_eval(array)
    numeric_array = np.array(array)
    return np.count(numeric_array)
def convert_and_min(array):
    # Convert each element in the array to a float
    array=ast.literal_eval(array)
    numeric_array = np.array(array)
    return np.min(numeric_array)

#%% load
clusters = pd.read_csv('clusters.csv')
istat_path = os.path.join(".","istat_microdata_2013.csv") #os.getcwd()
data = pd.read_csv(istat_path, delimiter=";")

#%% merge
merged_df = pd.merge(data, clusters[["segment","identi"]], left_on='id', right_on='identi', how='inner')
merged_df.drop(columns=['identi'], inplace=True)
merged_df.reset_index()

#%% clean
data_to_analyze=merged_df[["segment","normal_l","normal_g","normal_s",
                           "q_2_1","q_3_17",
                           "q_2_4_ric","q_2_7_class",
                           "reg_new","rip"]]
data_to_analyze["abitazione"]=merged_df["q_2_1"].map(
    {1:"casa unifamiliare",
     2:"casa plurifamiliare",
     3:"appartamento in edificio piccolo",
     4:"appartamento in edificio medio",
     5:"appartamento in edificio grande"})
data_to_analyze["isolamento"]=merged_df["q_3_17"].map(
    {1:"Ben isolata",
     2:"Abbastanza isolata",
     3:"Poco isolata",
     4:"Non isolata"})
data_to_analyze["age"]=merged_df["q_2_4_ric"]
data_to_analyze["age"]=data_to_analyze["age"].map(
    {1:[0,13],2:[14,23],3:[24,33],4:[34,43],5:[44,53],6:[54,63],7:[64,113],8:[113,150],9:[0,70]})
data_to_analyze['age'] = data_to_analyze['age'].apply(random_age_from_range)
data_to_analyze["area"]=merged_df["q_2_7_class"]
data_to_analyze["area"]=data_to_analyze["area"].map(
    {1:[10,19],2:[20,39],3:[40,59],4:[60,89],5:[90,119],6:[120,149],7:[150,400]})
data_to_analyze['area'] = data_to_analyze['area'].apply(random_age_from_range)
data_to_analyze["regione"]=merged_df["reg_new"].map(
    {10:"Piemonte", 20:"Valle d'Aosta",30:"Lombardia",41:"Bolzano",
     42:"Trento",50:"Veneto",60:"Friuli Venezia Giulia",70:"Liguria",
     80:"Emilia Romagna",90:"Toscana",100:"Umbria",110:"Marche",120:"Lazio",
     130:"Abruzzo",140:"Molise",150:"Campania",160:"Puglia",170:"Basilicata",
     180:"Calabria", 190:"Sicilia",200:"Sardegna"})
data_to_analyze["longitude"]=merged_df["reg_new"].map(
    {10:7.77, 20:7.31,30:8.95,41:11.35,
     42:11.12,50:12.33,60:12.65,70:8.93,
     80:11.35,90:11.26,100:12.38,110:13.51,120:12.48,
     130:13.39,140:14.66,150:14.29,160:16.86,170:15.80,
     180:16.58, 190:13.36,200:9.10})
data_to_analyze["latitude"]=merged_df["reg_new"].map(
    {10:45.45, 20:45.80,30:45.35,41:46.49,
     42:46.07,50:45.43,60:45.96,70:44.40,
     80:44.49,90:43.77,100:43.11,110:43.61,120:41.89,
     130:42.35,140:41.56,150:40.83,160:41.12,170:40.63,
     180:38.90, 190:38.11,200:39.21})
data_to_analyze["ripartizione"]=merged_df["rip"].map(
    {1:"Nord_Ovest",
     2:"Nord_Est",
     3:"Centrale",
     4:"Sud e Isole"})

#%% X and Y
Data_to_classify=data_to_analyze[["segment","abitazione",
                                  "isolamento","age",
                                  "area","latitude","longitude"]]

Data_to_classify["segment"][Data_to_classify["segment"]!=0]=1
sns.pairplot(Data_to_classify,hue="segment")
# Define feature columns and target column
X = Data_to_classify.drop(columns=["segment"])
y = Data_to_classify[["segment"]]

#%% Regression


# Define categorical and numerical columns
categorical_features = ['abitazione', 'isolamento']
numerical_features = ['area', 'age',"latitude","longitude"]
scaler = StandardScaler()
X[['age', 'area', 'latitude','longitude']] = scaler.fit_transform(X[['age', 'area', 'latitude','longitude']])


X = pd.get_dummies(X, columns=['abitazione', 'isolamento'], drop_first=True)

#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

#%%
from sklearn.metrics import classification_report, confusion_matrix

y_pred = clf.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#%%
from imblearn.combine import SMOTEENN


y[y!=0]=1
smote_enn = SMOTEENN()
# X_resampled, y_resampled = smote_enn.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clfim = RandomForestClassifier()
clfim.fit(X_train, y_train)
y_pred = clfim.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False, xticklabels=['Class 1', 'Class 2'], yticklabels=['Class 1', 'Class 2'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


#%% X and Y
Data_to_classify=data_to_analyze[["normal_l","normal_g","normal_s","abitazione",
                                  "isolamento","age",
                                  "area","latitude","longitude"]]


sns.pairplot(Data_to_classify,hue="abitazione",plot_kws={'s': 1})
# Define feature columns and target column
X = Data_to_classify.drop(columns=["normal_l","normal_g","normal_s"])
y = Data_to_classify[["normal_l","normal_g","normal_s"]]

#%% Regression


# Define categorical and numerical columns
categorical_features = ['abitazione', 'isolamento']
numerical_features = ['area', 'age',"latitude","longitude"]
scaler = StandardScaler()
X[['age', 'area', 'latitude','longitude']] = scaler.fit_transform(X[['age', 'area', 'latitude','longitude']])


X = pd.get_dummies(X, columns=['abitazione', 'isolamento'], drop_first=True)

#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

#%%
from sklearn.metrics import classification_report, confusion_matrix

y_pred = clf.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#%%
from imblearn.combine import SMOTEENN


y[y!=0]=1
smote_enn = SMOTEENN()
# X_resampled, y_resampled = smote_enn.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clfim = RandomForestClassifier()
clfim.fit(X_train, y_train)
y_pred = clfim.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False, xticklabels=['Class 1', 'Class 2'], yticklabels=['Class 1', 'Class 2'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()