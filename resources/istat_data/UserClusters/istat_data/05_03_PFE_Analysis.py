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

Probablities=pd.read_csv("probablities_only.csv")
Probablities=Probablities[Probablities["total_number"]>4]
Regression_data=Probablities[["age","area","prob0","prob1","prob2","prob3","region","type","isolation"]]
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
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming you have a DataFrame `df` with columns 'age' and 'area'
# Also, assuming age_th and area_th are your threshold values
age_th = 150   # Example threshold for age
area_th = 150  # Example threshold for area
rsquared_mean=0
# Define conditions
conditions = [
    (Regression_data['age'] < age_th) & (Regression_data['area'] < area_th),     # Condition for 0
    (Regression_data['age'] < age_th) & (Regression_data['area'] > area_th),     # Condition for 1
    (Regression_data['age'] > age_th) & (Regression_data['area'] < area_th),     # Condition for 2
    (Regression_data['age'] > age_th) & (Regression_data['area'] > area_th)      # Condition for 3
]

# Define choices for each condition
choices = [0, 1, 2, 3]

# Create the new column based on the conditions and choices
Regression_data['category'] = np.select(conditions, choices)
Regression_data=Regression_data[Regression_data['category']==0]
# Regression_data=Regression_data[Regression_data['category']!=3]
R=Regression_data.copy()
# Regression_data=Regression_data[Regression_data["isolation"]!="ALL"]
Regression_data['age']=Regression_data['age']/100
Regression_data['area']=Regression_data['area']/100

#%%
formula = """
prob0 ~ age + area + I(age**2) 
     
"""

# Perform backward elimination to find the best model
model0 = backward_elimination(Regression_data, 'prob0', formula)
# Summary of the model

# Evaluating the model using metrics
# 1. R-squared
r_squared = model0.rsquared
adjusted_r_squared = model0.rsquared_adj

# 2. AIC/BIC
aic = model0.aic
bic = model0.bic

# 3. RMSE
predictions = model0.predict(Regression_data)
p0=predictions
rmse = np.sqrt(mean_squared_error(Regression_data['prob0'], predictions))
# plt.figure()
# plt.scatter(Regression_data['prob0'], predictions)
# plt.show()
# print(f"R-squared: {r_squared}")
print(f"Adjusted R-squared: {adjusted_r_squared}")
# print(f"AIC: {aic}")
# print(f"BIC: {bic}")
print(f"RMSE: {rmse}")
rsquared_mean+=0.25*r_squared
#%%

formula = """
prob1 ~ age + area + I(age**2) 
"""

# Perform backward elimination to find the best model
model1 = backward_elimination(Regression_data, 'prob1', formula)
# Summary of the model

# Evaluating the model using metrics
# 1. R-squared
r_squared = model1.rsquared
adjusted_r_squared = model1.rsquared_adj

# 2. AIC/BIC
aic = model1.aic
bic = model1.bic

# 3. RMSE
predictions = model1.predict(Regression_data)
p1=predictions
rmse = np.sqrt(mean_squared_error(Regression_data['prob1'], predictions))
# plt.figure()
# plt.scatter(Regression_data['prob1'], predictions)
# plt.show()
# print(f"R-squared: {r_squared}")
print(f"Adjusted R-squared: {adjusted_r_squared}")
# print(f"AIC: {aic}")
# print(f"BIC: {bic}")
print(f"RMSE: {rmse}")
rsquared_mean+=0.25*r_squared
#%%

formula = """
prob2 ~  age + area + I(age**2) 

"""
# formula = """
# prob2 ~ age + area + I(age**2) + I(area**2) + age:area 


# Perform backward elimination to find the best model
model2 = backward_elimination(Regression_data, 'prob2', formula)
# Summary of the model
# Evaluating the model using metrics
# 1. R-squared
r_squared = model2.rsquared
adjusted_r_squared = model2.rsquared_adj

# 2. AIC/BIC
aic = model2.aic
bic = model2.bic

# 3. RMSE
predictions = model2.predict(Regression_data)
p2=predictions
rmse = np.sqrt(mean_squared_error(Regression_data['prob2'], predictions))
plt.figure()
plt.scatter(Regression_data['prob2'], predictions)
plt.show()
# print(f"R-squared: {r_squared}")
print(f"Adjusted R-squared: {adjusted_r_squared}")
# print(f"AIC: {aic}")
# print(f"BIC: {bic}")
print(f"RMSE: {rmse}")
rsquared_mean+=0.25*r_squared

#%%

formula = """
prob3 ~  age + area + I(age**2)  
        
"""

Regression_data=Regression_data
# Perform backward elimination to find the best model
model3 = backward_elimination(Regression_data, 'prob3', formula)
# Summary of the model
# Evaluating the model using metrics
# 1. R-squared
r_squared = model3.rsquared
adjusted_r_squared = model3.rsquared_adj

# 2. AIC/BIC
aic = model3.aic
bic = model3.bic

# 3. RMSE
predictions = model3.predict(Regression_data)
# predictions=1-p1-p2-p0
p3=predictions
rmse = np.sqrt(mean_squared_error(Regression_data['prob3'], predictions))
plt.figure()
plt.scatter(Regression_data['prob3'], predictions)
plt.show()
# print(f"R-squared: {r_squared}")
print(f"Adjusted R-squared: {adjusted_r_squared}")
# print(f"AIC: {aic}")
# print(f"BIC: {bic}")
print(f"RMSE: {rmse}")
rsquared_mean+=0.25*r_squared
#%%
print(rsquared_mean)
formula = """
prob3 ~ age + area + I(age**2)  + age:area 

        
"""
# formula = """
# prob1 ~ age + area + I(age**2) + I(area**2) + age:area 
#         + C(region) + C(type) + C(isolation) + age:C(region) 
#         + age:C(type) + age:C(isolation) + area:C(region) 
#         + area:C(type) + area:C(isolation)
        
# """

# # Perform backward elimination to find the best model
# model0 = backward_elimination(Regression_data, 'prob1', formula)
# # Summary of the model


# # Evaluating the model using metrics
# # 1. R-squared
# r_squared = model0.rsquared
# adjusted_r_squared = model0.rsquared_adj

# # 2. AIC/BIC
# aic = model0.aic
# bic = model0.bic

# # 3. RMSE
# predictions = model0.predict(Regression_data)
# p0=predictions
# rmse = np.sqrt(mean_squared_error(Regression_data['prob1'], predictions))
# plt.figure()
# plt.scatter(Regression_data['prob1'], predictions)
# plt.show()
# print(f"R-squared: {r_squared}")
# print(f"Adjusted R-squared: {adjusted_r_squared}")
# print(f"AIC: {aic}")
# print(f"BIC: {bic}")
# print(f"RMSE: {rmse}")
# rsquared_mean+=0.25*r_squared
# formula = """
# prob1 ~ age + area + I(age**2) + I(area**2) + age:area 
        
        
# # """

# # Perform backward elimination to find the best model
# model1 = backward_elimination(Regression_data, 'prob1', formula)
# # Summary of the model


# # Evaluating the model using metrics
# # 1. R-squared
# r_squared = model1.rsquared
# adjusted_r_squared = model1.rsquared_adj

# # 2. AIC/BIC
# aic = model1.aic
# bic = model1.bic

# # 3. RMSE
# predictions = model1.predict(Regression_data)
# p1=predictions
# rmse = np.sqrt(mean_squared_error(Regression_data['prob1'], predictions))
# plt.figure()
# plt.scatter(Regression_data['prob1'], predictions)
# plt.show()
# print(f"R-squared: {r_squared}")
# print(f"Adjusted R-squared: {adjusted_r_squared}")
# print(f"AIC: {aic}")
# print(f"BIC: {bic}")
# print(f"RMSE: {rmse}")
# rsquared_mean+=0.25*r_squared

# formula = """
# prob3 ~ age + area + I(age**2) + I(area**2) + age:area 
        
        
# """

# # Perform backward elimination to find the best model
# model3 = backward_elimination(Regression_data, 'prob3', formula)
# # Summary of the model

# # Evaluating the model using metrics
# # 1. R-squared
# r_squared = model3.rsquared
# adjusted_r_squared = model3.rsquared_adj

# # 2. AIC/BIC
# aic = model3.aic
# bic = model3.bic

# # 3. RMSE
# predictions = model3.predict(Regression_data)
# p3=predictions
# rmse = np.sqrt(mean_squared_error(Regression_data['prob3'], predictions))
# plt.figure()
# plt.scatter(Regression_data['prob3'], predictions)
# plt.show()

# # Printing evaluation metrics
# print(f"R-squared: {r_squared}")
# print(f"Adjusted R-squared: {adjusted_r_squared}")
# print(f"AIC: {aic}")
# print(f"BIC: {bic}")
# print(f"RMSE: {rmse}")
# rsquared_mean+=0.25*r_squared




# #%%



# formula = """
# prob2 ~ age + area  +  I(age**2)+I(area**2) + age:area 
        
        
# """

# # Perform backward elimination to find the best model
# model2 = backward_elimination(Regression_data, 'prob2', formula)
# # Summary of the model


# # Evaluating the model using metrics
# # 1. R-squared
# r_squared = model2.rsquared
# adjusted_r_squared = model2.rsquared_adj

# # 2. AIC/BIC
# aic = model2.aic
# bic = model2.bic

# # 3. RMSE
# predictions = model2.predict(Regression_data)

# # predictions=1-p0-p1-p3
# Regression_data['prob2pred']=predictions

# # Create the scatter plot with category as hue
# sns.scatterplot(data=Regression_data, x='prob2', y='prob2pred', hue='category', palette='viridis', s=100)

# # Add titles and labels for clarity
# plt.title('Scatter plot of prob2 vs prob2pred with Category as Hue')
# plt.xlabel('prob2 (Actual Values)')
# plt.ylabel('prob2pred (Predicted Values)')

# # Display the plot
# plt.show()
# # # predictions=model.predict(Regression_data)
# rmse = np.sqrt(mean_squared_error(Regression_data['prob2'], predictions))
# rsquared_mean+=0.25*r_squared
# print(rsquared_mean)
# # plt.figure()
# # # plt.xlim([0.10,0.5])
# # # plt.ylim([0.1,0.5])
# # plt.scatter(Regression_data['prob2'], predictions-Regression_data['prob2'])
# # plt.show()
# # print(f"R-squared: {r_squared}")
# # print(f"Adjusted R-squared: {adjusted_r_squared}")
# # print(f"AIC: {aic}")
# # print(f"BIC: {bic}")
# # print(f"RMSE: {rmse}")

# #%%
# # # Import necessary libraries
# # import pandas as pd
# # from sklearn.tree import DecisionTreeRegressor
# # from sklearn.model_selection import train_test_split
# # from sklearn.metrics import r2_score
# # from sklearn import tree

# # # Assume df is your pandas DataFrame containing the necessary columns
# # # X = Features (age, area, age_area), y = Target (prob2)
# # Regression_data=Regression_data[Regression_data['area']<150]
# # Regression_data=Regression_data[Regression_data['age']<100]
# # X = Regression_data[['age', 'area']]
# # y = Regression_data['prob2']

# # # Split the data into 80% training and 20% testing
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # # Initialize and fit the Decision Tree Regressor
# # treedata = DecisionTreeRegressor(random_state=42,max_depth=3)
# # treedata.fit(X_train, y_train)

# # # Make predictions on the test set
# # y_pred = treedata.predict(X_test)

# # # Evaluate the performance using R-squared
# # r2 = r2_score(y_test, y_pred)
# # print(f"R-squared score: {r2}")

# # # Optionally: You can compare predictions vs actual values by creating a DataFrame
# # comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

# # # Visualize the prediction vs actual values (optional)
# # import matplotlib.pyplot as plt

# # plt.scatter(y_test, y_pred)
# # plt.xlabel("Actual Values")
# # plt.ylabel("Predicted Values")
# # plt.title("Decision Tree: Actual vs Predicted")
# # plt.show()

# # # Plot the decision tree
# # plt.figure(figsize=(20,10))  # Set a large figure size for better readability
# # tree.plot_tree(treedata, feature_names=X.columns, filled=True, rounded=True)
# # plt.show()

#%%
# import pandas as pd
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # Sample DataFrame (replace with your actual DataFrame)
# # df = pd.DataFrame({
# #     'age': [20, 25, 30, 35],
# #     'area': [200, 250, 300, 350],
# #     'prob0': [0.1, 0.3, 0.5, 0.7],
# #     'isolamento': [0, 1, 2, 3]  # Use your actual isolamento data here
# # })

# # Create a 3D scatter plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Define a colormap (you can use other colormaps like 'plasma', 'inferno', etc.)
# cmap = plt.get_cmap('viridis')

# # Map specific isolamento values to colors (manual mapping)
# color_map = {0: 'red', 1: 'orange', 2: 'blue', 3: 'green'}

# # Create a new column in the DataFrame to store the colors
# Regression_data['color'] = Regression_data['isolation'].map(color_map)
# # Scatter plot: use 'age' for x, 'area' for y, 'prob0' for z, and 'isolamento' for color
# sc = ax.scatter(Regression_data['age'], Regression_data['area'], Regression_data['prob0'], c=Regression_data['color'])

# # Label the axes
# ax.set_xlabel('Age')
# ax.set_ylabel('Area')
# ax.set_zlabel('Prob0')

# # Add a color bar
# cbar = plt.colorbar(sc)
# cbar.set_label('isolation')

# # Show the plot
# plt.show()

#%%
Regression_data["probpred0"]=p0
Regression_data["probpred1"]=p1
Regression_data["probpred2"]=p2
Regression_data["probpred3"]=p3
Regression_data["age"]=Regression_data["age"]*100

#%%
import matplotlib.pyplot as plt

# Setting up the plot
plt.figure(figsize=(12, 16), facecolor="white")
ax = plt.gca()
ax.set_facecolor("white")

# Customizations
plt.grid(color="lightgray", linestyle="--", linewidth=0.5)

# Loop through unique area values and plot lineplots accordingly
areas = Regression_data["area"].unique()

colors = ['#4169E1', '#50C878', '#FFBF00', '#D50032']  # Assign colors to prob groups

# Loop over the prob and probpred variables
for i, (prob, probpred) in enumerate(zip(["prob0", "prob1", "prob2", "prob3"], 
                                         ["probpred0", "probpred1", "probpred2", "probpred3"])):
    
    # Scatter plot for each probability
    plt.scatter(Regression_data["age"], Regression_data[prob], color=colors[i], alpha=0.5,
                marker="x", label=f"{prob} (Scatter)")

    # Lineplot for each unique area and probpred
    for k, area in enumerate(areas):
        subset = Regression_data[Regression_data["area"] == area]
        line_label = f"Area={area} - {probpred}"  # Create a label for the area and probpred
        
        # Plot the line for each area
        line = plt.plot(subset["age"], subset[probpred], linestyle="--", linewidth=2, 
                        color=colors[i])

        # Add text next to the line indicating the area
        # Positioning the text slightly to the right of the first data point
        x_pos = subset["age"].iloc[len(subset) *(i+1)//5]  # Middle point of the line
        y_pos = subset[probpred].iloc[len(subset) *(i+1)//5]  # Middle point of the line
        if k in [0,4]:
            plt.text(x_pos, y_pos, f"area={int(area*100//1)} m$^2$", color="black", fontsize=16, ha='left', va='center',
                 bbox=dict(facecolor=colors[i], alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))
        
# Adding labels and grid
plt.xlabel("Dwelling's Age", fontsize=20)
plt.ylabel("Probability", fontsize=20)


# Create a custom legend for the areas
legend_labels = ['Cluster 4', 'Cluster 3', 'Cluster 2', 'Cluster 1']  # Desired order for clusters
colors_legend = {
    'A': '#4169E1',
    'B': '#50C878',
    'C': '#FFBF00',
    'D': '#D50032'
}
plt.xticks(rotation=0,fontsize=20)
plt.yticks(rotation=0,fontsize=20)
# Create custom legend handles for areas (these are color swatches corresponding to areas)
handles = [plt.Rectangle((0, 0), 1, 1, color=colors_legend[key]) for key in ['D', 'C', 'B', 'A']]

# Add legend for areas separately
plt.legend(handles, legend_labels, loc='upper left', fontsize=20)

# Show grid and plot
plt.grid(True)
plt.show()



#%%
import matplotlib.pyplot as plt

# Define colors for scatter points, one for each plot
colors = [             '#4169E1',
             '#50C878',
             '#FFBF00',
            '#D50032'] 

# Create a 2x2 grid for subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 12), facecolor="white")
fig.subplots_adjust(hspace=0.3, wspace=0.3)  # Adjust spacing between subplots

# Iterate over the grid and create scatter plots
for i, ax in enumerate(axes.flat):  # Enumerate through axes in the grid
    prob = f"prob{i}"
    probpred = f"probpred{i}"
    
    # Scatter plot
    ax.scatter(Regression_data[prob], Regression_data[probpred], color=colors[i], alpha=0.7,marker="x")
    
    # Set title for each subplot
    ax.set_title(f"Cluster {i+1}", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16, labelcolor='black')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1)

# Set overarching x and y labels
fig.text(0.5, 0.04, 'Actual Proportion', ha='center', fontsize=16)
fig.text(0.04, 0.5, 'Predicted Probability', va='center', rotation='vertical', fontsize=16)

# Show the plot
plt.show()

#%%
import matplotlib.pyplot as plt

# Define colors for scatter points, one for each plot
colors = ['#4169E1', '#50C878', '#FFBF00', '#D50032']

# Create a single figure
fig, ax = plt.subplots(figsize=(10, 10), facecolor="white")

# Iterate through the four clusters and add scatter points
for i, color in enumerate(colors):
    prob = f"prob{i}"
    probpred = f"probpred{i}"
    
    # Overlay scatter plot for each cluster
    ax.scatter(Regression_data[prob], Regression_data[probpred], color=color, alpha=0.7, label=f"Cluster {i+1}",marker="x")

# Add dashed line y = x
ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1, label="y = x")

# Set title and labels
ax.set_title("", fontsize=18)
ax.set_xlabel('Actual Proportion', fontsize=20)
ax.set_ylabel('Predicted Probability', fontsize=20)

# Customize ticks
ax.tick_params(axis='both', which='major', labelsize=16, labelcolor='black')

# Set limits
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# Add legend
ax.legend(fontsize=20, loc='upper left')

# Show the plot
plt.show()


#%%
import matplotlib.pyplot as plt


colors = [             '#4169E1',
             '#50C878',
             '#FFBF00',
            '#D50032'] 


fig, axes = plt.subplots(2, 2, figsize=(12, 12), facecolor="white")
fig.subplots_adjust(hspace=0.3, wspace=0.3)  # Adjust spacing between subplots
Regression_data['age_group'] = pd.cut(
    Regression_data['age'],
    bins=[-float('inf'), 23, 43, 68, float('inf')],  # Define bins for the age ranges
    labels=['After 1990', '1970-1990', '1945-1970', 'before 1945'],  # Define corresponding labels
    right=True  # Include the right boundary in each bin
)
age_classes = Regression_data["age_group"].unique()
for i, ax in enumerate(axes.flat):  
    prob = f"prob{i}"
    probpred = f"probpred{i}"
    

    ax.scatter(Regression_data["area"]*100, Regression_data[probpred], color=colors[i], alpha=0.5,
                marker="x")
    
    ax.set_title(f"Cluster {i+1}", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16, labelcolor='black')
    for k, age_class in enumerate(age_classes):
        subset = Regression_data[Regression_data["age_group"] == age_class]
        line_label = f"age_group={age_class}"  # Create a label for the area and probpred
        

        mean_probpred0_by_area = subset.groupby('area')[probpred].mean().reset_index()
        mean_probpred0_by_area['area']=mean_probpred0_by_area['area']*100
        sns.lineplot(
            data=mean_probpred0_by_area,
            x='area',
            y=probpred,
            linestyle="--",
            color="Black",
            marker='o',
            ax=ax  
            )


 



plt.show()

#%%
import matplotlib.pyplot as plt
import seaborn as sns

colors = ['#4169E1', '#50C878', '#FFBF00', '#D50032']

fig, axes = plt.subplots(2, 2, figsize=(12, 12), facecolor="white")
fig.subplots_adjust(hspace=0.3, wspace=0.3)  # Adjust spacing between subplots

Regression_data['age_group'] = pd.cut(
    Regression_data['age'],
    bins=[-float('inf'), 23, 43, 68, float('inf')],  # Define bins for the age ranges
    labels=['After 1990', '1970-1990', '1945-1970', 'before 1945'],  # Define corresponding labels
    right=True  # Include the right boundary in each bin
)

age_classes = Regression_data["age_group"].unique()

for i, ax in enumerate(axes.flat):
    prob = f"prob{i}"
    probpred = f"probpred{i}"

    ax.scatter(Regression_data["area"] * 100, Regression_data[probpred], color=colors[i], alpha=0.5,
               marker="x")

    ax.set_title(f"Cluster {i + 1}", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=16, labelcolor='black')

    for k, age_class in enumerate(age_classes):
        subset = Regression_data[Regression_data["age_group"] == age_class]
        mean_probpred0_by_area = subset.groupby('area')[probpred].mean().reset_index()
        mean_probpred0_by_area['area'] = mean_probpred0_by_area['area'] * 100

        # Plot the dashed line
        line = sns.lineplot(
            data=mean_probpred0_by_area,
            x='area',
            y=probpred,
            linestyle="--",
            color="darkgray",
            marker='o',
            ax=ax
        )

        # Annotate the line with the label
        if not mean_probpred0_by_area.empty:  # Ensure there is data to annotate
            last_point = mean_probpred0_by_area.iloc[1]
            ax.annotate(
                f"{age_class}",
                xy=(last_point['area'], last_point[probpred]*0.95),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=16,
                color="black"
                # backgroundcolor="white"
            )
        ax.set_xlabel('', fontsize=1)
        ax.set_ylabel('', fontsize=1)

fig.text(0.5, 0.04, 'Dwelling Area (m$^2$)', ha='center', fontsize=20)
fig.text(0.04, 0.5, 'Predicted Probability', va='center', rotation='vertical', fontsize=20)

plt.show()

#%%
import matplotlib.pyplot as plt

# Create the 2x2 figure
fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor="white")
fig.subplots_adjust(hspace=0.3, wspace=0.3)  # Adjust spacing between subplots

# Assign colors for the different clusters
colors = ['#4169E1', '#50C878', '#FFBF00', '#D50032']  

# Unique areas
areas = Regression_data["area"].unique()

# Loop over clusters and create one subplot for each
for i, (prob, probpred) in enumerate(zip(["prob0", "prob1", "prob2", "prob3"], 
                                         ["probpred0", "probpred1", "probpred2", "probpred3"])):
    ax = axes[i // 2, i % 2]  # Select the subplot in the 2x2 grid
    ax.set_facecolor("white")
    ax.grid(color="lightgray", linestyle="--", linewidth=0.5)

    # Scatter plot for probabilities
    ax.scatter(Regression_data["age"], Regression_data[prob], color=colors[i], alpha=0.5,
               marker="x", label=f"{prob} (Scatter)")

    # Lineplot for each unique area
    for k, area in enumerate(areas):
        subset = Regression_data[Regression_data["area"] == area]
        if subset.empty:
            continue

        # Plot the line for each area
        ax.plot(subset["age"], subset[probpred], linestyle="--", linewidth=2, 
                label=f"Area {area * 100:.0f} m²", alpha=0.9,color=colors[i])

        # Optionally, add a label near the lines
        midpoint = len(subset) // 2
        if midpoint > 0:
            ax.text(subset["age"].iloc[midpoint], subset[probpred].iloc[midpoint], 
                    f"{int(area * 100)} m²", fontsize=10, color="black", alpha=0.7)

    # Customize axes and title
    ax.set_title(f"Cluster {i + 1}", fontsize=16, color=colors[i])
    ax.set_xlabel("Dwelling's Age", fontsize=14)
    ax.set_ylabel("Probability", fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)

# Show the complete figure
plt.show()
