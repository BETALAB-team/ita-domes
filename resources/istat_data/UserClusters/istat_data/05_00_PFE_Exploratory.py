# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 12:53:11 2024

@author: khajmoh18975
"""

import networkx as nx
import matplotlib.pyplot as plt
import os
import pandas as pd
import ast
import statsmodels.formula.api as smf
from patsy import dmatrices
from sklearn.metrics import mean_squared_error
import numpy as np
import numpy as np
import random
import seaborn as sns
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

Probablities=pd.read_csv("probablities_of_housing.csv")
Probablities=Probablities[Probablities["total_number"]>4]
Probablities=Probablities[Probablities["region"]!="ALL"]
Probablities=Probablities[Probablities["type"]!="ALL"]
Probablities=Probablities[Probablities["isolation"]!="ALL"]
Regression_data=Probablities[["age","area","region","type","isolation","prob0","prob1","prob2","prob3"]]
# sns.pairplot(Regression_data,hue="isolation",plot_kws={'s': 4})
# plt.show()
# sns.pairplot(Regression_data,hue="region")
# plt.show()
# sns.pairplot(Regression_data,hue="type")
# plt.show()

def backward_elimination(data, response, formula):
    # Fit the model initially with all variables
    model = smf.ols(formula=formula, data=data).fit()
    while True:
        # Get the p-values for all variables
        pvalues = model.pvalues
        
        # Check the max p-value (remove if greater than 0.05)
        max_pval = pvalues.max()
        if max_pval > 0.05:
            # Get the name of the variable with the highest p-value
            excluded_var = pvalues.idxmax()
            
            # If this is not the intercept, update the formula
            if excluded_var != 'Intercept':
                if ':' not in excluded_var:
                    # Remove main effect and all interactions involving it
                    var_name = excluded_var.split('[')[0]  # Remove any category level info, e.g., C(region)[T.North]
                    formula = formula.replace(f' + {excluded_var}', '').replace(f'{excluded_var} + ', '')
                    
                    # Remove interactions involving this variable
                    formula = formula.replace(f'{var_name}:', '').replace(f':{var_name}', '')
                    print(f"Removing {excluded_var} and all its interactions due to high p-value ({max_pval})")
                else:
                    # Remove only the interaction term
                    formula = formula.replace(f' + {excluded_var}', '').replace(f'{excluded_var} + ', '')
                    print(f"Removing interaction {excluded_var} due to high p-value ({max_pval})")
                
                # Refit the model with the updated formula
                model = smf.ols(formula=formula, data=data).fit()
            else:
                break
        else:
            break
    
    # Return the final model and its summary
    return model
# Full quadratic model formula (including interactions and quadratic terms)
formula = """
prob2 ~ age + area + I(age**2) + I(area**2) + age:area 
         + C(isolation)  
        + age:C(isolation) 
"""

# Perform backward elimination to find the best model
model = backward_elimination(Regression_data, 'prob1', formula)
# Summary of the model
print(model.summary())

# Evaluating the model using metrics
# 1. R-squared
r_squared = model.rsquared
adjusted_r_squared = model.rsquared_adj

# 2. AIC/BIC
aic = model.aic
bic = model.bic

# 3. RMSE
predictions = model.predict(Regression_data)
rmse = np.sqrt(mean_squared_error(Regression_data['prob1'], predictions))

# Printing evaluation metrics
print(f"R-squared: {r_squared}")
print(f"Adjusted R-squared: {adjusted_r_squared}")
print(f"AIC: {aic}")
print(f"BIC: {bic}")
print(f"RMSE: {rmse}")