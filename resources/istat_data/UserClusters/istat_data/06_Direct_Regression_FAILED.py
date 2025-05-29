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

def random_age_from_range(age_range):
    if isinstance(age_range, list) and len(age_range) == 2:
        return random.randint(age_range[0], age_range[1])
    return None

import statsmodels.api as sm
#%%
#%% load
clusters = pd.read_csv(os.path.join(".","clusters.csv"))
elets = pd.read_csv(os.path.join(".","results_grins.csv"), delimiter=",")
istat_path = os.path.join(".","istat_microdata_2013.csv") #os.getcwd()
data = pd.read_csv(istat_path, delimiter=";")
data["elet"]=elets["appliances_el_kWh"]

#%% merge
merged_df = pd.merge(data, clusters[["segment","normal_l","normal_g","normal_s","identi"]], left_on='id', right_on='identi', how='inner')
merged_df.drop(columns=['identi'], inplace=True)
merged_df.reset_index()

#%% clean
data_to_analyze=merged_df[["segment","normal_l","normal_g","normal_s",
                           "q_2_1","q_3_17",
                           "q_2_4_ric","q_2_7_class",
                           "reg_new","rip","q_3_11A","q_3_11B","q_3_11C","q_9_1_ric_2","elet"]]
data_to_analyze["riscaldament_fascia_A"]=merged_df["q_3_11A"]
data_to_analyze["riscaldament_fascia_B"]=merged_df["q_3_11B"]
data_to_analyze["riscaldament_fascia_C"]=merged_df["q_3_11C"]
data_to_analyze["riscaldament"]=merged_df["q_3_11A"]+merged_df["q_3_11B"]+merged_df["q_3_11C"]
data_to_analyze["raffrescament_fascia_A"]=merged_df["q_5_8A"]
data_to_analyze["raffrescament_fascia_B"]=merged_df["q_5_8B"]
data_to_analyze["raffrescament_fascia_C"]=merged_df["q_5_8C"]
data_to_analyze["raffrescament"]=merged_df["q_5_8A"]+merged_df["q_5_8B"]+merged_df["q_5_8C"]
data_to_analyze["elettricita"]=merged_df["elet"]

# Additional processing and statistical analysis...

# Perform statistical analysis using regression models
formula = "riscaldament ~ C(segment) + C(abitazione) + C(isolamento) + age + area + degree_days + I(age**2) + I(area**2) + C(seg_cas) + C(seg_rip)"
model = sm.ols(formula, data=data_to_analyze).fit()
print(model.summary())

# Visualizing results
plt.figure(figsize=(10, 6))
sns.boxplot(x='segment', y='riscaldament', data=data_to_analyze)
plt.title('Boxplot of Riscaldamento by Segment')
plt.xlabel('Segment')
plt.ylabel('Riscaldamento')
plt.show()

# Perform Tukey HSD test
from statsmodels.stats.multicomp import pairwise_tukeyhsd
data_to_analyze['group_combination'] = data_to_analyze['segment'].astype(str) + "-" + data_to_analyze['abitazione'].astype(str) + "-" + data_to_analyze['ripartizione'].astype(str)
tukey_results = pairwise_tukeyhsd(endog=data_to_analyze['riscaldament'], groups=data_to_analyze['group_combination'], alpha=0.05)
print(tukey_results)
