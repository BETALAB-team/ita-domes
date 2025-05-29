# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 17:09:45 2024

@author: khajmoh18975
"""

#%%
import pandas as pd
import os
import re
Path=os.path.join(".","Significant_groups.csv")
DF_groups=pd.read_csv(Path)



# Create the new DataFrame with unique new_group and new_group_r values
result = (
    DF_groups.groupby("combined_factors")[["new_group", "new_group_r"]]
    .agg(lambda x: list(x.unique()))
    .reset_index()
)

# Define a function that splits on _ but not after "Nord"
def safe_split(value):
    # This regex splits on underscores but avoids splitting after "Nord"
    return re.split(r'(?<!Nord)_', value)

# Apply the safe_split function to the 'combined_factors' column
result[["part1", "part2", "part3"]] = (
    result["combined_factors"]
    .apply(safe_split)
    .apply(lambda x: x + [None] * (3 - len(x)))  # Ensure exactly 3 parts
    .apply(pd.Series)
)

# Rename columns for clarity
result.columns = [
    "combined_factors", 
    "unique_new_groups", 
    "unique_new_group_r", 
    "part1", 
    "part2", 
    "part3"
]



result['unique_new_groups'] = result['unique_new_groups'].apply(lambda x: x[0])
unique_values = result['unique_new_groups'].unique()
mapping = {value: f"Group{i+1}" for i, value in enumerate(unique_values)}
result['unique_new_groups'] = result['unique_new_groups'].map(mapping)


result['unique_new_group_r'] = result['unique_new_group_r'].apply(lambda lst: [x for x in lst if x != 'Zero'])
result['unique_new_group_r'] = result['unique_new_group_r'].apply(lambda x: x[0])
unique_values = result['unique_new_group_r'].unique()
mapping = {value: f"Group{i+1}" for i, value in enumerate(unique_values)}
result['unique_new_group_r'] = result['unique_new_group_r'].map(mapping)
#%%
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# Assuming your DataFrame is named DF_groups
# Formula: response ~ categorical_variables + numerical_variables + interaction_terms
DF_groups['Cooling'] = DF_groups['raffrescament'].apply(lambda x: 1 if x != 0 else 0)
formula = 'riscaldament ~ C(abitazione) + C(isolamento) + C(segment) + C(ripartizione) + age + area + C(abitazione):C(isolamento) + C(abitazione):C(segment) + C(abitazione):C(ripartizione) + C(isolamento):C(segment) + C(isolamento):C(ripartizione) + C(segment):C(ripartizione) + age:area+C(Cooling):C(isolamento) +C(Cooling)'

# Fit the ANOVA model
model = ols(formula, data=DF_groups).fit()

# Perform the ANOVA
anova_results = anova_lm(model, typ=2)  # Type 2 sum of squares, commonly used for unbalanced designs

# Display the results
print(anova_results)

#%%
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# Assuming your DataFrame is named DF_groups
# Formula: response ~ categorical_variables + numerical_variables + interaction_terms
DF_group_c=DF_groups[DF_groups["raffrescament"]!=0]
formula = 'raffrescament ~ C(abitazione) + C(isolamento) + C(segment) + C(ripartizione) + age + area + C(abitazione):C(isolamento) + C(abitazione):C(segment) + C(abitazione):C(ripartizione) + C(isolamento):C(segment) + C(isolamento):C(ripartizione) + C(segment):C(ripartizione) + age:area '

# Fit the ANOVA model
model = ols(formula, data=DF_group_c).fit()

# Perform the ANOVA
anova_results = anova_lm(model, typ=2)  # Type 2 sum of squares, commonly used for unbalanced designs

# Display the results
print(anova_results)
#%%
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'DF_groups' is your DataFrame and it contains 'riscaldament' and 'segment' columns
plt.figure(figsize=(10, 6))  # Set the size of the plot

# Create the boxplot
sns.boxplot(x='ripartizione', y='raffrescament', data=DF_group_c)

# Adding titles and labels
plt.title('Boxplot of riscaldament by Segment')
plt.xlabel('Segment')
plt.ylabel('Riscaldament')

# Display the plot
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability if necessary
plt.show()

#%%
import numpy as np
Path=os.path.join(".","new_elet.csv")
DF_groupse=pd.read_csv(Path)
np.random.seed(42)
random_numbers = np.random.randint(0, 20, size=len(DF_groupse))

DF_groupse["elettricita"]=DF_groupse["elettricita"]


import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# Assuming your DataFrame is named DF_groups
# Formula: response ~ categorical_variables + numerical_variables + interaction_terms
DF_groupse['Cooling'] = DF_groupse['raffrescament'].apply(lambda x: 1 if x != 0 else 0)
formula = 'elettricita ~ C(abitazione) + C(isolamento) + C(segment) + C(ripartizione)  + C(abitazione):C(isolamento) + C(abitazione):C(segment) + C(abitazione):C(ripartizione) + C(isolamento):C(segment) + C(isolamento):C(ripartizione) + C(segment):C(ripartizione)  +C(Cooling):C(isolamento) +C(Cooling)'

# Fit the ANOVA model
model = ols(formula, data=DF_groupse).fit()

# Perform the ANOVA
anova_results = anova_lm(model, typ=2)  # Type 2 sum of squares, commonly used for unbalanced designs
print(model.summary())
# Display the results
print(anova_results)
#%%
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'DF_groups' is your DataFrame and it contains 'riscaldament' and 'segment' columns
plt.figure(figsize=(10, 6))  # Set the size of the plot

# Create the boxplot
sns.boxplot(x='segment', y='elettricita', data=DF_groupse)

# Adding titles and labels
plt.title('Boxplot of riscaldament by Segment')
plt.xlabel('Segment')
plt.ylabel('Riscaldament')

# Display the plot
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability if necessary
plt.show()


#%%
# Calculate the average of 'riscaldament' for each unique string in 'your_column_name'
average_riscaldament = DF_groups.groupby('new_group')['riscaldament'].mean()

# Sort the unique strings by their average 'riscaldament'
sorted_strings = average_riscaldament.sort_values().index

# Create a mapping of the sorted strings to "Group {i}" labels
string_to_group = {value: f"Group {i}" for i, value in enumerate(sorted_strings, start=1)}

# Apply the mapping to the column
DF_groups['new_group'] = DF_groups['new_group'].map(string_to_group)

#%%
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Generate a color palette in shades of red
num_groups = DF_groups['new_group'].nunique()
red_shades = sns.color_palette("Reds", num_groups)

# Sort groups and map them to colors
sorted_groups = sorted(DF_groups['new_group'].unique())
color_mapping = {group: red_shades[i] for i, group in enumerate(sorted_groups)}

# Plot the boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(
    x='new_group',
    y='riscaldament',
    data=DF_groups,
    order=sorted_groups,
    palette=color_mapping,
    showfliers=False  # Hides the outliers
)

# Add labels and title
plt.xlabel('Groups', fontsize=14)
plt.ylabel('Riscaldament', fontsize=14)
plt.title('Boxplot of Riscaldament by Group (Shades of Red)', fontsize=16)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.tight_layout()
plt.show()

#%%
import pandas as pd

# Create a copy of DF_groups
DF_plots = DF_groups.copy()

# Remove rows where 'isolamento' is "nan" (string) or NaN (actual missing value)
DF_plots = DF_plots[~DF_plots['isolamento'].isin(["nan"]) & DF_plots['isolamento'].notna()]
import seaborn as sns
import matplotlib.pyplot as plt
# Map the values in the 'isolamento' column
isolation_mapping = {
    "Poco isolata": "Not sufficiently insulated",
    "Non isolata": "Not sufficiently insulated",
    "Ben isolata": "Sufficiently insulated",
    "Abbastanza isolata": "Sufficiently insulated"
}

# Apply the mapping
DF_plots['isolamento'] = DF_plots['isolamento'].map(isolation_mapping)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
DF_plots['new_groups']=DF_plots['new_group']

#%%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Assuming DF_plots is your DataFrame
# Convert columns to strings before concatenating
DF_plots['new_groups'] = DF_plots['new_groups'].astype(str)
DF_plots['isolamento'] = DF_plots['isolamento'].astype(str)

# Create a new column that concatenates new_groups and isolamento
DF_plots['group_isolamento'] = DF_plots['new_groups'] + '_' + DF_plots['isolamento']

# Create a list of positions for the x-axis (21 positions)
positions = []
x_labels = []

# For each group (1 to 7), we need to insert 2 entries:
# one for "Not sufficiently insulated" and one for "Sufficiently insulated"
for n in range(1, 8):
    # Positions (3n-2) for 'Not sufficiently insulated'
    positions.append(3*n - 2)
    x_labels.append(f"Group {n} - Not sufficiently insulated")
    
    # Positions (3n-1) for 'Sufficiently insulated'
    positions.append(3*n - 1)
    x_labels.append(f"Group {n} - Sufficiently insulated")
    
    # Every 3rd place will be empty, so we don't add anything

# Plotting
plt.figure(figsize=(16, 8))

# Use a custom palette for the boxplot (optional, but useful for visualization)
palette = sns.color_palette("Set2", n_colors=7)

# Now, plot the boxplot using the 'group_isolamento' as the categorical x-axis
sns.boxplot(data=DF_plots, x='group_isolamento', y='riscaldament', palette=palette)

# Set custom x-ticks with positions, ensuring they match the expected group layout
plt.xticks(ticks=positions, labels=x_labels, rotation=45)

# Title and labels for clarity
plt.title('Boxplot of Riscaldament by Group and Insulation')
plt.xlabel('Group and Insulation Status')
plt.ylabel('Riscaldament')

# Adjust layout for readability
plt.tight_layout()

# Show the plot
plt.show()


#%%
# Assuming the DataFrame `DF_plots` is loaded and contains the required columns

# Adjust the legend to show only the two categories: "Well insulated" and "Not well-insulated"
# Set up the plot style
sns.set(style="whitegrid")

# Create the boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(
    x="new_groups",
    y="riscaldament",
    hue="group_isolamento",
    data=DF_plots,
    palette=["#F4D4B0", "#E9AC72"]
)

# Customize plot
plt.xlabel(None, fontsize=14)
plt.ylabel("Daily on-time of heating System (hours)", fontsize=14)
plt.legend(title="Insulation Type", title_fontsize=12, fontsize=10, labels=["Well insulated", "Not well-insulated"])
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.ylim(0, 20)  # Adjust y-axis if needed

# Show plot
plt.tight_layout()
plt.show()

#%%
DF_groupse["elettricita"]=DF_groupse["elettricita"]/DF_groupse["area"]
#%%
# Perform Tukey's HSD on the three-way interaction
from statsmodels.stats.multicomp import pairwise_tukeyhsd

DF_groupse["locationAbition"]=DF_groupse["ripartizione"]+"-"+DF_groupse["abitazione"]
data_to_cool = DF_groupse[DF_groupse['raffrescament'] != 0]
tukey_interaction_r = pairwise_tukeyhsd(endog=DF_groupse['elettricita'],
                                      groups=DF_groupse['locationAbition'],
                                      alpha=0.05)

# Print the Tukey HSD results for the three-way interaction
print("\nTukey HSD Results for Segment:Abitazione:Ripartizione Interaction:")
print(tukey_interaction_r)


#%%
import pandas as pd
DF_groupse['new_group_e'] = DF_groupse['locationAbition']
# Assuming `tukey_interaction` is the Tukey HSD result object and `DF_groupse` is the original DataFrame
num=80 
for i in range(num):
   
    #   Step 1: Convert Tukey HSD result to DataFrame

    tukey_df = pd.DataFrame(tukey_interaction_r.summary().data[1:], columns=tukey_interaction_r.summary().data[0])
    rejected_count = tukey_df['reject'].sum()  # Count of True values (rejected)
    not_rejected_count = (~tukey_df['reject']).sum()  # Count of False values (not rejected)
    if(not_rejected_count==0):
        break
    #   Step 2: Find the row with the largest adjusted p-value (which is the least significant difference)
    largest_pval_row = tukey_df.loc[tukey_df['p-adj'].idxmax()]
    
    # Step 3: Extract the two groups with the largest p-value
    group1 = largest_pval_row['group1']
    group2 = largest_pval_row['group2']
    
    # Step 4: Create the 'new_group' column by copying 'segment_abitazione_ripartizione' values

    unique_values_count = DF_groupse['new_group_e'].nunique()
    print(f"Number of unique values in 'new_group_e': {unique_values_count} || {rejected_count},{not_rejected_count} || p_value of change: {largest_pval_row}")
    # Step 5: Update the 'new_group' column where the values match group1 or group2
    DF_groupse['new_group_e'] = DF_groupse['new_group_e'].replace([group1, group2], f"HSDR_group{i+1}")

    # Step 6: Print the first few rows of the updated DataFrame to check the result
    tukey_interaction_r = pairwise_tukeyhsd(endog=DF_groupse['elettricita'],
                                          groups=DF_groupse['new_group_e'],
                                          alpha=0.05)
    
    #%%
import seaborn as sns
import matplotlib.pyplot as plt

# Calculate the median of 'riscaldament' for each 'new_group'
median_values = DF_groupse.groupby('new_group_e')['elettricita'].median().sort_values()

# Create a sorted list of new_group based on the median values
sorted_groups = median_values.index

# Set up the plot
plt.figure(figsize=(10, 6))

# Create the boxplot, sorting the 'new_group' by the median values
sns.boxplot(x='new_group_e', y='elettricita', data=DF_groupse, order=sorted_groups)

# Set plot labels and title
plt.xlabel('New Group')
plt.ylabel('raffrescament')
plt.title('Boxplot of Riscaldament Grouped by New Group (Sorted by Median)')

# Show the plot
plt.xticks(rotation=45)  # Optional: Rotate x-axis labels for better readability if needed
plt.show()
result_dict = (
    DF_groupse.groupby('new_group_e')['locationAbition']
    .apply(lambda x: list(x.unique()))
    .to_dict()
)

print(result_dict)

# Remove rows where 'raffrescament' equals 0

#%%
#%%
# Update 'Significant_group' values in DF_groupse for matching indexes in data_to_cool
DF_groups['new_group_r']="Zero"
DF_groups.loc[DF_groupse.index.isin(data_to_cool.index), 'new_group_r'] = \
    data_to_cool['new_group_r'].reindex(DF_groupse.index)

for category, group in DF_groupse.groupby('new_group_r'):
    print(f"Category: {category}")
    print(group['new_group'].value_counts())
    print("\n")
    
    
    #%%
    # Group by 'new_group' and calculate mean and std for 'riscaldament'
group_stats = DF_groups.groupby('new_group')['riscaldament'].agg(['mean', 'std'])

# Print the statistics for each group
print(group_stats)
#%%
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Sort groups by the mean value of 'riscaldament' (or 'elettricita')
group_means = DF_groupse.groupby('new_group_e')['elettricita'].mean().sort_values()

# Sort the groups based on their mean 'elettricita' values
sorted_groups = group_means.index.tolist()

# Create a mapping for the new group names (Group 1, Group 2, etc.)
group_name_mapping = {group: f"Group {i+1}" for i, group in enumerate(sorted_groups)}

# Apply the new group names to the dataframe
DF_groupse['group_name'] = DF_groupse['new_group_e'].map(group_name_mapping)

# Generate a color palette in shades of red based on the number of unique groups
num_groups = len(sorted_groups)
red_shades = sns.color_palette("Reds", num_groups)

# Create a color mapping for the groups
color_mapping = {group_name_mapping[group]: red_shades[i] for i, group in enumerate(sorted_groups)}

# Plot the boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(
    x='group_name',  # Use the new group names for the x-axis
    data=DF_groupse,
    order=[f"Group {i+1}" for i in range(num_groups)],  # Ensure the groups appear in the sorted order
    palette=color_mapping,
    showfliers=False  # Hides the outliers
)

# Add labels and title
plt.xlabel('Groups', fontsize=14)
plt.ylabel('Riscaldament', fontsize=14)
plt.title('Boxplot of Riscaldament by Group (Shades of Red)', fontsize=16)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.tight_layout()
plt.show()
#%%
# Group the data by 'new_group_e' and get unique 'locationAbition' values for each group
group_location_dict = DF_groupse.groupby('group_name')['locationAbition'].unique().to_dict()

# Print the resulting dictionary
print(group_location_dict)

#%%
# Perform Tukey's HSD on the three-way interaction
from statsmodels.stats.multicomp import pairwise_tukeyhsd
DF_groups['abitazione'] = DF_groups['abitazione']
DF_groups["locationAbitionUser"]=DF_groups["ripartizione"]+"-"+DF_groups["abitazione"]+"-"+(DF_groups["segment"].astype(str))
data_to_cool = DF_groupse[DF_groups['raffrescament'] != 0]
tukey_interaction_r = pairwise_tukeyhsd(endog=DF_groups['riscaldament'],
                                      groups=DF_groups['locationAbitionUser'],
                                      alpha=0.05)

# Print the Tukey HSD results for the three-way interaction
print("\nTukey HSD Results for Segment:Abitazione:Ripartizione Interaction:")
print(tukey_interaction_r)

import pandas as pd
DF_groups['new_group_e'] = DF_groups['locationAbitionUser']
# Assuming `tukey_interaction` is the Tukey HSD result object and `DF_groupse` is the original DataFrame
num=80 
for i in range(num):
   
    #   Step 1: Convert Tukey HSD result to DataFrame

    tukey_df = pd.DataFrame(tukey_interaction_r.summary().data[1:], columns=tukey_interaction_r.summary().data[0])
    rejected_count = tukey_df['reject'].sum()  # Count of True values (rejected)
    not_rejected_count = (~tukey_df['reject']).sum()  # Count of False values (not rejected)
    if(not_rejected_count==0):
        break
    #   Step 2: Find the row with the largest adjusted p-value (which is the least significant difference)
    largest_pval_row = tukey_df.loc[tukey_df['p-adj'].idxmax()]
    
    # Step 3: Extract the two groups with the largest p-value
    group1 = largest_pval_row['group1']
    group2 = largest_pval_row['group2']
    
    # Step 4: Create the 'new_group' column by copying 'segment_abitazione_ripartizione' values

    unique_values_count = DF_groups['new_group_e'].nunique()
    print(f"Number of unique values in 'new_group_e': {unique_values_count} || {rejected_count},{not_rejected_count} || p_value of change: {largest_pval_row}")
    # Step 5: Update the 'new_group' column where the values match group1 or group2
    DF_groups['new_group_e'] = DF_groups['new_group_e'].replace([group1, group2], f"HSDR_group{i+1}")

    # Step 6: Print the first few rows of the updated DataFrame to check the result
    tukey_interaction_r = pairwise_tukeyhsd(endog=DF_groups['riscaldament'],
                                          groups=DF_groups['new_group_e'],
                                          alpha=0.05)
#%%
# Sort groups by the mean value of 'riscaldament' (or 'elettricita')
group_means = DF_groups.groupby('new_group')['riscaldament'].mean().sort_values()

# Sort the groups based on their mean 'elettricita' values
sorted_groups = group_means.index.tolist()

# Create a mapping for the new group names (Group 1, Group 2, etc.)
group_name_mapping = {group: f"Group H{i+1}" for i, group in enumerate(sorted_groups)}

# Apply the new group names to the dataframe
DF_groups['group_name'] = DF_groups['new_group'].map(group_name_mapping)
# Group the data by 'new_group_e' and get unique 'locationAbition' values for each group
group_location_dict = DF_groups.groupby('group_name')['locationAbitionUser'].unique().to_dict()

# Print the resulting dictionary
print(group_location_dict)

#%%
# Generate a color palette in shades of red based on the number of unique groups
num_groups = len(sorted_groups)
red_shades = sns.color_palette("Reds", num_groups)

# Create a color mapping for the groups
color_mapping = {group_name_mapping[group]: red_shades[i] for i, group in enumerate(sorted_groups)}

# Plot the boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(
    x='group_name',  # Use the new group names for the x-axis
    y='riscaldament',  # Change 'elettricita' if you want to plot 'riscaldament' instead
    data=DF_groups,
    order=[f"Group H{i+1}" for i in range(num_groups)],  # Ensure the groups appear in the sorted order
    palette=color_mapping,
    showfliers=False  # Hides the outliers
)

# Add labels and title
plt.xlabel('Groups', fontsize=14)
plt.ylabel('Riscaldament', fontsize=14)
plt.title('Boxplot of Riscaldament by Group (Shades of Red)', fontsize=16)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.tight_layout()
plt.show()

#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Create DF_plots from the specified columns and clean isolamento
DF_plots = DF_groups[['group_name', 'riscaldament', 'isolamento']].copy()
DF_plots["new_group"]=DF_plots["group_name"]
DF_plots = DF_plots[~DF_plots['isolamento'].isin([None, 'nan'])]  # Remove rows with nan or "nan"

# Step 2: Map isolamento to the new categories
mapping = {
    'Non isolata': 'Not well-insulated',
    'Poco isolata': 'Not well-insulated',
    'Ben isolata': 'Well insulated',
    'Abbastanza isolata': 'Well insulated'
}
DF_plots['isolamento'] = DF_plots['isolamento'].map(mapping)

# Step 3: Calculate average riscaldament by new_group and sort by it
avg_riscaldament = DF_plots.groupby('new_group')['riscaldament'].mean().sort_values()
sorted_groups = avg_riscaldament.index

# Step 4: Set up the boxplot with proper ordering and coloring
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

# Custom color mapping based on riscaldament averages
color_palette = sns.color_palette("Oranges", n_colors=len(sorted_groups))
group_colors = {group: color_palette[i] for i, group in enumerate(sorted_groups)}

# Create a grouped boxplot
boxplot = sns.boxplot(
    data=DF_plots,
    x='new_group',
    y='riscaldament',
    hue='isolamento',
    order=sorted_groups,
    hue_order=['Well insulated', 'Not well-insulated'],
    palette=[group_colors[group] for group in sorted_groups] * 2,
    showfliers=False
)

# Adjust plot aesthetics
plt.xlabel("")
plt.tick_params(axis='both', which='major', labelsize=18)
plt.ylabel("Daily on-time of heating System (hours)",fontsize=20)
plt.legend(title="Insulation Type", loc="upper left",fontsize=20)
plt.tight_layout()

# Show plot
plt.show()

#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Create DF_plots from the specified columns and clean isolamento
DF_plots = DF_groupse[['new_group_e', 'elettricita', 'Cooling','segment','area',"locationAbition"]].copy()
# Sort groups by the mean value of 'riscaldament' (or 'elettricita')
group_means = DF_plots.groupby('new_group_e')['elettricita'].mean().sort_values()

# Sort the groups based on their mean 'elettricita' values
sorted_groups = group_means.index.tolist()

# Create a mapping for the new group names (Group 1, Group 2, etc.)
group_name_mapping = {group: f"Group E{i+1}" for i, group in enumerate(sorted_groups)}
DF_plots['new_group_e'] = DF_plots['new_group_e'].map(group_name_mapping)
# DF_plots['elettricita_spec']=DF_plots['elettricita']
# DF_plots['elettricita']=DF_plots['elettricita']*DF_plots['area']

#%%
DF_plots = DF_plots[~DF_plots['Cooling'].isin([None, 'nan'])]  # Remove rows with nan or "nan"

# Step 2: Map isolamento to the new categories
mapping = {
    0: 'Cooling System Not Used',
    1: 'Cooling System Used'
}
DF_plots['Cooling'] = DF_plots['Cooling'].map(mapping)

# Step 3: Calculate average riscaldament by new_group and sort by it
avg_riscaldament = DF_plots.groupby('new_group_e')['elettricita'].mean().sort_values()
sorted_groups = avg_riscaldament.index

# Step 4: Set up the boxplot with proper ordering and coloring
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

# Custom color mapping based on riscaldament averages
color_palette = sns.color_palette("YlOrBr", n_colors=len(sorted_groups))
group_colors = {group: color_palette[i] for i, group in enumerate(sorted_groups)}

# Create a grouped boxplot
boxplot = sns.boxplot(
    data=DF_plots,
    x='new_group_e',
    y='elettricita',
    hue='Cooling',
    order=sorted_groups,
    hue_order=['Cooling System Not Used', 'Cooling System Used'],
    palette=[group_colors[group] for group in sorted_groups] * 2,
    showfliers=False
)

# Adjust plot aesthetics
plt.xlabel("")
plt.ylabel("Annual Electricity Spending (€)",fontsize=20)

plt.tick_params(axis='both', which='major', labelsize=18)

plt.legend(title="Cooling System", loc="upper left",fontsize=16)
plt.tight_layout()

# Show plot
plt.show()

#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Create a new dataframe with necessary columns
DF_plotse = DF_groupse[['segment', 'elettricita',"area","age"]].copy()
DF_plotse['segment']= "Cluster " + (DF_plotse['segment']+1).astype(str)
# Step 2: Calculate average elettricita by segment and sort by it
avg_elettricita = DF_plotse.groupby('segment')['elettricita'].mean().sort_values()
sorted_segments = avg_elettricita.index

# Step 3: Set up the boxplot with proper ordering and coloring
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

# Custom color mapping based on elettricita averages
color_palette = sns.color_palette("YlOrBr", n_colors=len(sorted_segments))
segment_colors = {segment: color_palette[i] for i, segment in enumerate(sorted_segments)}

# Create the boxplot
boxplot = sns.boxplot(
    data=DF_plotse,
    x='segment',
    y='elettricita',
    order=sorted_segments,
    palette=[segment_colors[segment] for segment in sorted_segments],
    showfliers=False
)

# Adjust plot aesthetics
plt.title("Boxplot of Elettricita by Segment")
plt.xlabel("Segment")
plt.ylabel("Elettricita")
plt.tight_layout()

# Show plot
plt.show()
#%%
DF_plotse["elettricita"]=DF_plotse["elettricita"]/DF_plotse["area"]
#%%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 
# Define color mapping for clusters
all_colors = {
    'Cluster 1': '#FFBF00',
    'Cluster 2': '#50C878',
    'Cluster 3': '#D50032',
    'Cluster 4': '#4169E1'
}
    # 'Cluster 1': '#4169E1',
    # 'Cluster 2': '#50C878',
    # 'Cluster 3': '#FFBF00',
    # 'Cluster 4': '#D50032'
# Create a mapping for the segment to the clusters and the corresponding colors
segment_to_cluster = {
    0: 'centroid_1',
    1: 'centroid_2',
    2: 'centroid_3',
    3: 'centroid_4'
}

# Map segment to cluster centroid
DF_plotse['cluster'] = DF_plotse['segment']

# Set up the boxplot with color mapping
plt.figure(figsize=(10, 6))
sns.boxplot(
    x='segment', 
    y='elettricita', 
    data=DF_plotse, 
    palette=[all_colors[cluster] for cluster in DF_plotse['segment'].unique()],
    showfliers=False
)

# Update x-axis labels to Cluster 1, Cluster 2, Cluster 3, Cluster 4
plt.xticks(ticks=[0, 1, 2, 3], labels=['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4'], fontsize=20)
plt.yticks(fontsize=20)

plt.xlabel(' ', fontsize=20)
plt.ylabel("Annual Electricity Spending (€)",fontsize=20)
# No title
# Show plot
plt.show()


#%%
import seaborn as sns
import matplotlib.pyplot as plt

# Set the figure size and style
plt.figure(figsize=(12, 8))
sns.set_style("whitegrid")

# Create the boxplot
sns.boxplot(data=DF_plots, x="segment", y="elettricita", palette="Set2",    showfliers=False)

# Add titles and labels
plt.title("Boxplot of 'elettricita_spec' by 'segment'", fontsize=16)
plt.xlabel("Segment", fontsize=14)
plt.ylabel("Elettricità Spec (Euro/metr2)", fontsize=14)

# Rotate x-axis labels if necessary
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)

# Show the plot
plt.tight_layout()
plt.show()

#%%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Sample DataFrame setup (adjust as per the actual structure)
DF_plotse=DF_plotse[DF_plotse["area"]<150]
# Create a new column 'age_class' based on the age ranges
def classify_age(age):
    if age > 68:
        return "Before 1945"
    elif 43 < age <= 68:
        return "1945-1970"
    elif 23 < age <= 43:
        return "1970-1990"
    else:
        return "After 1990"

DF_plotse['age_class'] = DF_plotse['age'].apply(classify_age)

# Define color palette for age classes
age_class_colors = {
    "Before 1945": "#4169E1",  # Royal Blue
    "1945-1970": "#50C878",   # Emerald
    "1970-1990": "#FFBF00",   # Amber
    "After 1990": "#D50032"   # Crimson
}

# Set up the scatterplot
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=DF_plotse,
    x='area',
    y='elettricita',
    hue='age_class',
    palette=age_class_colors,
    s=10  # Set point size for better visibility
)

# Customize plot
plt.xlabel('Area', fontsize=14)
plt.ylabel('Elettricita', fontsize=14)
plt.legend(title='Age Class', fontsize=12, title_fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Show plot
plt.show()

#%%
# Create a boxplot for elettricita grouped by age_class
plt.figure(figsize=(10, 6))
sns.boxplot(
    data=DF_plotse,
    x='age_class',
    y='elettricita',
    palette=age_class_colors
)

# Customize plot
plt.xlabel('Age Class', fontsize=14)
plt.ylabel('Elettricita', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Show plot
plt.show()

#%%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Sample DataFrame setup (adjust as per the actual structure)
DF_plotse=DF_plotse[DF_plotse["area"]<150]
# Create a new column 'age_class' based on the age ranges
def classify_age(age):
    if age > 68:
        return "Before 1945"
    elif 43 < age <= 68:
        return "1945-1970"
    elif 23 < age <= 43:
        return "1970-1990"
    else:
        return "After 1990"

DF_plotse['age_class'] = DF_plotse['age'].apply(classify_age)
age_class_order = ["Before 1945", "1945-1970", "1970-1990", "After 1990"]
# Group the 'area' column into 4 equal parts and create a new column 'area_class'
labels = []
area_classes = pd.qcut(DF_plotse['area'], q=4)

# Create custom labels like "XX-YY"
for i, interval in enumerate(area_classes.cat.categories):
    if i == 0:  # First quartile (Smaller than max)
        labels.append(f"Smaller than {int(interval.right)} m$^2$")
    elif i == 3:  # Last quartile (Larger than min)
        labels.append(f"Larger than {int(interval.left)} m$^2$")
    else:  # Middle quartiles (XX-YY)
        labels.append(f"{int(interval.left)}m$^2$-{int(interval.right)} m$^2$")

DF_plotse['area_class'] = area_classes.cat.rename_categories(labels)

# Create a boxplot grouped by both 'age_class' and 'area_class'
plt.figure(figsize=(14, 8))
sns.boxplot(
    data=DF_plotse,
    x='age_class',
    y='elettricita',
    hue='area_class',
    palette='Set2',
    order=age_class_order,
    showfliers=False
)

# Customize plot
plt.xlabel('Age Class', fontsize=0)
plt.ylabel("Annual Electricity Spending (€)", fontsize=20)
plt.legend(title='Area Class', fontsize=16, title_fontsize=18,bbox_to_anchor=(1, 1),loc="upper left")
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
# Show plot
plt.show()

#%%
DF_plotsh = DF_groups[['segment', 'riscaldament',"area","age"]].copy()

DF_plotsh=DF_plotsh[DF_plotsh["area"]<150]
# Create a new column 'age_class' based on the age ranges
def classify_age(age):
    if age > 68:
        return "Before 1945"
    elif 43 < age <= 68:
        return "1945-1970"
    elif 23 < age <= 43:
        return "1970-1990"
    else:
        return "After 1990"

DF_plotsh['age_class'] = DF_plotsh['age'].apply(classify_age)
age_class_order = ["Before 1945", "1945-1970", "1970-1990", "After 1990"]
# Group the 'area' column into 4 equal parts and create a new column 'area_class'
labels = []
area_classes = pd.qcut(DF_plotsh['area'], q=4)

# Create custom labels like "XX-YY"
for i, interval in enumerate(area_classes.cat.categories):
    if i == 0:  # First quartile (Smaller than max)
        labels.append(f"Smaller than {int(interval.right)} m$^2$")
    elif i == 3:  # Last quartile (Larger than min)
        labels.append(f"Larger than {int(interval.left)} m$^2$")
    else:  # Middle quartiles (XX-YY)
        labels.append(f"{int(interval.left)}m$^2$-{int(interval.right)} m$^2$")

DF_plotsh['area_class'] = area_classes.cat.rename_categories(labels)

# Create a boxplot grouped by both 'age_class' and 'area_class'
plt.figure(figsize=(14, 8))
sns.boxplot(
    data=DF_plotsh,
    x='age_class',
    y='riscaldament',
    hue='area_class',
    palette='Set2',
    order=age_class_order,
    showfliers=False
)

# Customize plot
plt.xlabel('Age Class', fontsize=0)
plt.ylabel("Annual Electricity Spending (€)", fontsize=20)
plt.legend(title='Area Class', fontsize=16, title_fontsize=18,bbox_to_anchor=(1, 1),loc="upper left")
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
# Show plot
plt.show()

plt.figure(figsize=(14, 8))
sns.boxplot(
    data=DF_plotsh,
    x='age_class',
    y='riscaldament',
    palette='Set2',
    order=age_class_order,
    showfliers=False
)

# Customize plot
plt.xlabel('Age Class', fontsize=0)
plt.ylabel("Annual Electricity Spending (€)", fontsize=20)
plt.legend(title='Area Class', fontsize=16, title_fontsize=18,bbox_to_anchor=(1, 1),loc="upper left")
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
# Show plot
plt.show()

#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataframe from the CSV file
data = pd.read_csv("new_elet.csv")

# Add a new column to indicate whether the cooling system is used or not
data['Cooling System'] = data['raffrescament'].apply(lambda x: 'Cooling System Used' if x != 0 else 'Cooling System Not Used')

# Sort groups by the median value of 'elettricita'
group_medians = data.groupby('new_group_e')['elettricita'].median().sort_values()
group_order = group_medians.index

# Rename groups based on their order
group_rename_map = {group: f"Group E{i+1}" for i, group in enumerate(group_order)}
data['new_group_e'] = data['new_group_e'].replace(group_rename_map)

# Create the boxplot without outliers
plt.figure(figsize=(10, 6))
sns.boxplot(
    x='new_group_e', 
    y='elettricita', 
    hue='Cooling System', 
    data=data, 
    palette=['#FAD7A0', '#F5B041'],
    order=[f"Group E{i+1}" for i in range(len(group_order))],
    showfliers=False
)

# Customize plot appearance
plt.xticks(fontsize=18)
plt.yticks(fontsize=16)
plt.xlabel("", fontsize=16)
plt.ylabel("Annual Appliance Consumption (kWh)", fontsize=18)
plt.title("", fontsize=1)
plt.legend(title="Cooling System", fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.tight_layout()
plt.show()
plt.savefig("Elec.svg", format="svg", bbox_inches="tight")

#%%

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# Update font settings to use Times New Roman
rcParams['font.family'] = 'Times New Roman'

# Create the boxplot without outliers
plt.figure(figsize=(10, 6))
sns.boxplot(
    x='new_group_e', 
    y='elettricita', 
    hue='Cooling System', 
    data=data, 
    palette=['#A3C1E3', '#1B365D'],
    order=[f"Group E{i+1}" for i in range(len(group_order))],
    showfliers=False
)

# Customize plot appearance
plt.xticks(fontsize=18)
plt.yticks(fontsize=16)
plt.xlabel("", fontsize=16)
plt.ylabel("Annual Appliance Consumption (kWh)", fontsize=18)
plt.title("", fontsize=1)
plt.legend(title="Cooling System", fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.tight_layout()
plt.show()

# Save the plot as SVG
plt.savefig("Elec.svg", format="svg", bbox_inches="tight")
#%%

#%%
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# Update font settings to use Times New Roman
rcParams['font.family'] = 'Times New Roman'
# Compute the group order based on median 'riscaldament'
group_medians = DF_plots.groupby('new_groups')['riscaldament'].median().sort_values()
group_order_h = group_medians.index
DF_plots['new_group'] = DF_plots['new_group'].str.replace(r'Group (\d+)', r'Group H\1', regex=True)

# Create the boxplot without outliers
plt.figure(figsize=(10, 6))
sns.boxplot(
    x='new_group', 
    y='riscaldament', 
    hue='isolamento',
    data=DF_plots, 
    palette=['#F4D4B0', '#E9AC72'],
    order=[f"Group H{i+1}" for i in range(len(group_order_h))],
    showfliers=False
)

# Customize plot appearance
plt.xticks(fontsize=18)
plt.yticks(fontsize=16)
plt.xlabel("", fontsize=16)
plt.ylabel("Daily On-Time of Heating System", fontsize=18)
plt.title("", fontsize=1)
plt.legend(title="Insulation Type", fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.tight_layout()
plt.show()

# Save the plot as SVG
plt.savefig("Heating.svg", format="svg", bbox_inches="tight")