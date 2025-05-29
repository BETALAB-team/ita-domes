import pandas as pd
import numpy as np
from matplotlib.ticker import MaxNLocator
#%% Data definition
data = {
    'region': [
        'Piemonte', 'Valle d\'Aosta', 'Lombardia', 'Veneto', 'Friuli Venezia Giulia', 'Liguria', 
        'Emilia Romagna', 'Toscana', 'Umbria', 'Marche', 'Lazio', 'Abruzzo', 'Molise', 
        'Campania', 'Puglia', 'Basilicata', 'Calabria', 'Sicilia', 'Sardegna', 'Trento', 'All'
    ],
    'centroid_1': [
        np.array([0.88, 0.7, 0.47]), np.array([0.47, 0.51, 0.76]), np.array([0.81, 0.69, 0.63]),
        np.array([0.63, 0.52, 0.6]), np.array([0.82, 0.65, 0.38]), np.array([0.89, 0.68, 0.48]),
        np.array([0.88, 0.69, 0.57]), np.array([0.46, 0.53, 0.74]), np.array([0.81, 0.68, 0.4]),
        np.array([0.65, 0.76, 0.53]), np.array([0.83, 0.68, 0.64]), np.array([0.72, 0.69, 0.64]),
        np.array([0.53, 0.55, 0.55]), np.array([0.63, 0.65, 0.58]), np.array([0.43, 0.58, 0.76]),
        np.array([0.45, 0.56, 0.75]), np.array([0.79, 0.66, 0.3]), np.array([0.85, 0.68, 0.56]),
        np.array([0.66, 0.55, 0.53]), np.array([0.68, 0.53, 0.52]), np.array([0.82, 0.67, 0.39])
    ],
    'centroid_2': [
        np.array([0.8, 0.63, 0.64]), np.array([0.75, 0.66, 0.75]), np.array([0.47, 0.55, 0.75]),
        np.array([0.87, 0.68, 0.6]), np.array([0.88, 0.69, 0.58]), np.array([0.67, 0.5, 0.51]),
        np.array([0.75, 0.59, 0.43]), np.array([0.67, 0.55, 0.51]), np.array([0.49, 0.5, 0.72]),
        np.array([0.87, 0.67, 0.52]), np.array([0.66, 0.57, 0.53]), np.array([0.42, 0.6, 0.75]),
        np.array([0.85, 0.71, 0.17]), np.array([0.43, 0.61, 0.76]), np.array([0.65, 0.55, 0.59]),
        np.array([0.83, 0.69, 0.59]), np.array([0.65, 0.49, 0.65]), np.array([0.6, 0.66, 0.58]),
        np.array([0.48, 0.59, 0.72]), np.array([0.59, 0.46, 0.75]), np.array([0.46, 0.56, 0.76])
    ],
    'centroid_3': [
        np.array([0.49, 0.52, 0.75]), np.array([0.87, 0.68, 0.49]), np.array([0.89, 0.66, 0.46]),
        np.array([0.43, 0.57, 0.76]), np.array([0.48, 0.55, 0.74]), np.array([0.78, 0.68, 0.64]),
        np.array([0.42, 0.63, 0.74]), np.array([0.86, 0.69, 0.51]), np.array([0.85, 0.66, 0.62]),
        np.array([0.64, 0.45, 0.63]), np.array([0.87, 0.68, 0.49]), np.array([0.85, 0.67, 0.45]),
        np.array([0.45, 0.6, 0.75]), np.array([0.8, 0.68, 0.58]), np.array([0.85, 0.69, 0.59]),
        np.array([0.66, 0.54, 0.54]), np.array([0.65, 0.44, 0.57]), np.array([0.86, 0.68, 0.56]),
        np.array([0.85, 0.67, 0.61]), np.array([0.63, 0.53, 0.52]), np.array([0.65, 0.53, 0.57])
    ],

    'centroid_4': [
        np.array([0.67, 0.5, 0.49]), np.array([0.69, 0.5, 0.5]), np.array([0.68, 0.51, 0.52]),
        np.array([0.79, 0.65, 0.41]), np.array([0.67, 0.53, 0.56]), np.array([0.49, 0.51, 0.76]),
        np.array([0.58, 0.46, 0.71]), np.array([0.77, 0.6, 0.79]), np.array([0.55, 0.72, 0.6]),
        np.array([0.44, 0.56, 0.77]), np.array([0.47, 0.54, 0.76]), np.array([0.62, 0.48, 0.62]),
        np.array([0.85, 0.69, 0.55]), np.array([0.84, 0.65, 0.53]), np.array([0.83, 0.6, 0.31]),
        np.array([0.84, 0.68, 0.22]), np.array([0.81, 0.69, 0.56]), np.array([0.66, 0.55, 0.31]),
        np.array([0.76, 0.55, 0.53]), np.array([0.39, 0.62, 0.75]), np.array([0.87, 0.67, 0.61])
    ],
    'CHI_score': [
        564.0796208, 366.1435893, 848.5216332, 590.38369, 413.7479621, 445.1079343, 540.6877933,
        486.0784643, 376.5885301, 369.4238465, 602.0637018, 374.5111265, 395.7942806, 523.6417486,
        574.8404778, 389.5139546, 439.2642336, 371.7924961, 517.3149361, 482.4888431, 10129.96908
    ]
}
dfchi = pd.DataFrame(data)
import pandas as pd

# Data for the first plot: Cluster using PCA Scores (WCSS vs N Clusters)
data1 = {
    'Number of Clusters': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    'Within Cluster Sum of Squares': [1250, 950, 580, 510, 430, 405, 305, 285, 265]  # Approximate values
}
df1 = pd.DataFrame(data1)

# Data for the second plot: Explained Variance by Component (up to 3 components)
data2 = {
    'Number of Components': [1, 2, 3],
    'Percentage of Cumulative Explained Variance': [82.5, 93.8, 98.5]  # Approximate values
}
df2 = pd.DataFrame(data2)

# Data for the third plot: Explained Variance by Component (up to 4 components)
data3 = {
    'Number of Components': [1, 2, 3, 4],
    'Percentage of Cumulative Explained Variance': [75.5, 89.0, 94.6, 97.2]  # Approximate values
}
df3 = pd.DataFrame(data3)

df_bar_one = pd.DataFrame({
    'cluster': ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4'],
    'houselods': [7163,4801,4080,1811],
    'people': [13051,17555,9279,3773]
})


#%%
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator

# Define font sizes for easy adjustment
label_fontsize = 22
tick_fontsize = 18
title_fontsize = 22

# Create a figure with two subplots side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Define plot settings
plot_params = {
    "color": "black",
    "marker": "o",
    "linewidth": 2
}

# Plot first graph (df2)
axes[1].tick_params(axis='both', which='major', labelsize=tick_fontsize)
axes[1].plot(df2['Number of Components'], df2['Percentage of Cumulative Explained Variance'], **plot_params)
axes[1].set_facecolor('white')
axes[1].grid(True, color='lightgray', linestyle='-', linewidth=0.5)
axes[1].set_xlabel(df2.columns[0], fontsize=label_fontsize)
axes[0].set_ylabel("Cumulative Explained Variance [%]", fontsize=label_fontsize)
axes[1].set_ylim(60, 100)
axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))

# Plot second graph (df3)
axes[0].tick_params(axis='both', which='major', labelsize=tick_fontsize)
axes[0].plot(df3['Number of Components'], df3['Percentage of Cumulative Explained Variance'], **plot_params)
axes[0].set_facecolor('white')
axes[0].grid(True, color='lightgray', linestyle='-', linewidth=0.5)
axes[0].set_xlabel(df3.columns[0], fontsize=label_fontsize)
axes[0].set_ylim(60, 100)
axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))

# Add (a) and (b) below the graphs
axes[0].text(0.5, -0.20, "(a)", fontsize=title_fontsize, transform=axes[0].transAxes, ha='center')
axes[1].text(0.5, -0.20, "(b)", fontsize=title_fontsize, transform=axes[1].transAxes, ha='center')

# Adjust layout
plt.tight_layout()
plt.show()
plt.savefig("cumulative_explained_variance.svg", format="svg", bbox_inches="tight")


#%% Line Charts
import matplotlib.pyplot as plt

# University of Padova color theme (University Red)
unipd_red = '#000000'

# Set font to Times New Roman for all plots
plt.rcParams['font.family'] = 'Times New Roman'

# Create a figure and axis for the first plot (df1)
fig1, ax1 = plt.subplots()
ax1.tick_params(axis='both', which='major', labelsize=16)
ax1.plot(df1['Number of Clusters'], df1['Within Cluster Sum of Squares'], color=unipd_red, marker='o')
ax1.set_facecolor('white')
ax1.grid(True, color='lightgray', linestyle='-', linewidth=0.5)
ax1.set_xlabel(df1.columns[0],fontsize=16)  # Use first column as x-label
ax1.set_ylabel(df1.columns[1],fontsize=16)  # Use second column as y-label
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.show()

# Create a figure and axis for the second plot (df2)
fig2, ax2 = plt.subplots()
ax2.tick_params(axis='both', which='major', labelsize=16)
ax2.plot(df2['Number of Components'], df2['Percentage of Cumulative Explained Variance'], color=unipd_red, marker='o')
ax2.set_facecolor('white')
ax2.grid(True, color='lightgray', linestyle='-', linewidth=0.5)
ax2.set_xlabel(df2.columns[0],fontsize=16)  # Use first column as x-label
ax2.set_ylabel(df2.columns[1],fontsize=16)  # Use second column as y-label
ax2.set_ylim(60, 100)

ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

plt.show()

# Create a figure and axis for the third plot (df3)
fig3, ax3 = plt.subplots()
ax3.tick_params(axis='both', which='major', labelsize=16)
ax3.plot(df3['Number of Components'], df3['Percentage of Cumulative Explained Variance'], color=unipd_red, marker='o')
ax3.set_facecolor('white')
ax3.grid(True, color='lightgray', linestyle='-', linewidth=0.5)
ax3.set_xlabel(df3.columns[0],fontsize=16)  # Use first column as x-label
ax3.set_ylabel(df3.columns[1],fontsize=16)  # Use second column as y-label
ax3.set_ylim(60, 100)

ax3.xaxis.set_major_locator(MaxNLocator(integer=True))

plt.show()

# #%% bar charts

# # Normalize the 'houselods' and 'people' columns by dividing by the sum of each column
df_bar_one['normalized_houselods'] = df_bar_one['houselods'] / df_bar_one['houselods'].sum()
df_bar_one['normalized_people'] = df_bar_one['people'] / df_bar_one['people'].sum()
# Set custom colors for each cluster
colors = {
    'A': '#4169E1',
    'B': '#50C878',
    'C': '#FFBF00',
    'D': '#D50032'
}

# Create a figure and axis
fig, ax = plt.subplots(figsize=(8, 6))
# Plot the 100% stacked bar chart with custom colors
normalized_data = pd.DataFrame({
    'households': df_bar_one['normalized_houselods'],
    'people': df_bar_one['normalized_people']
}).T
bars = normalized_data.plot(kind='bar', stacked=True, ax=ax, 
                            color=[colors['A'], colors['B'], colors['C'], colors['D']],width=0.4)

# Add percentage labels inside the corresponding areas
for i, bar in enumerate(bars.patches):
    height = bar.get_height()
    if height > 0:  # Only add text if the bar has height (non-zero)
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + height / 2, 
                f'{height*100:.1f}%', ha='center', va='center', color='white', fontsize=16)


ax.set_ylabel('Proportion of Total', fontsize=16)
ax.set_ylim(0, 1)

# Set the x-axis labels to cluster names
plt.xticks(rotation=0,fontsize=16)
plt.yticks(rotation=0,fontsize=16)
# Set background color
ax.set_facecolor('white')
legend_labels = ['Cluster 4', 'Cluster 3', 'Cluster 2', 'Cluster 1']  # Desired order
handles = [plt.Rectangle((0, 0), 1, 1, color=colors[key]) for key in ['D', 'C', 'B', 'A']]  # Match colors with order

ax.legend(handles, legend_labels, loc='lower center', fontsize=16)
# Add a legend with the corresponding colors for each cluster
# ax.legend(['Cluster 1', 'Cluster 2','Cluster 3','Cluster 4'], loc='upper center', fontsize=16)

# Show the plot
plt.subplots_adjust(right=100)
plt.tight_layout()
plt.show()

#%% CHI

import matplotlib.pyplot as plt
import numpy as np

# Custom colors for 'All' region centroids
all_colors = {
    'centroid_1': '#4169E1',
    'centroid_2': '#50C878',
    'centroid_3': '#FFBF00',
    'centroid_4': '#D50032'
}

# Create a figure with two subplots (side by side)
fig, ax = plt.subplots(2, 1, figsize=(7.5,20))

# Set background color for both subplots
# Set figure background color to white
fig.patch.set_facecolor('white')
for a in ax:
    a.set_facecolor('lightgray')

# First subplot (x, y plot)
for i, row in dfchi.iterrows():
    for j in range(1, 5):
        centroid = row[f'centroid_{j}']
        
        
        # X and Y values from centroid array
        x, y = centroid[2], centroid[0]
        
        # Color based on the "region" and custom colors for 'All'
        if row['region'] == 'All':
            color = all_colors[f'centroid_{j}']
        else:
            color = 'darkgray'
        
        # Size of the point, proportional to the square root of CHI_score
        size = row['CHI_score']**0.9/10  # Scaling the size for better visibility
        
        # Plot the point on the first axis
        ax[0].scatter(x, y, c=color, s=size, alpha=1)

# Set labels and title for first subplot
ax[0].set_ylabel('Normalized PC1',fontsize=14)


# Second subplot (x, z plot)
for i, row in dfchi.iterrows():
    for j in range(1, 5):
        centroid = row[f'centroid_{j}']
        
        # X and Z values from centroid array (third value)
        x, z = centroid[2], centroid[1]  # x = third value, y = second value for this plot
        
        # Color based on the "region" and custom colors for 'All'
        if row['region'] == 'All':
            color = all_colors[f'centroid_{j}']
        else:
            color = 'darkgray'
        size = row['CHI_score']**0.9/10
        # Size of the point, proportional to the square root of CHI_score
    # Scaling the size for better visibility
        
        # Plot the point on the second axis
        ax[1].scatter(x, z, c=color, s=size, alpha=1)

# Set labels and title for second subplot
ax[1].set_ylabel('Normalized PC2',fontsize=14)
ax[1].set_xlabel('Normalized Education',fontsize=14)

# Set gridlines for both subplots
for a in ax:
    a.grid(True, color='white', linestyle='-', linewidth=0.5)

# Show the plot
plt.tight_layout()
plt.show()

#%%
import matplotlib.pyplot as plt
import numpy as np

# Custom colors for 'All' region centroids
all_colors = {
    'centroid_1': '#4169E1',
    'centroid_2': '#50C878',
    'centroid_3': '#FFBF00',
    'centroid_4': '#D50032'
}

# Function to add blurry effect to lines
def plot_blurry_lines(ax, x_coords, y_coords, color, base_alpha=0.15, blur_iterations=25, max_offset=0.05):
    for _ in range(blur_iterations):
        # Generate random offsets for x and y coordinates
        offsets = np.random.uniform(-max_offset, max_offset, size=len(x_coords))
        x_blur = np.array(x_coords) + offsets
        y_blur = np.array(y_coords) + offsets
        
        # Calculate alpha inversely proportional to the offset magnitude
        offset_magnitude = np.abs(offsets)
        alpha = base_alpha * (1 - offset_magnitude / max_offset)  # Normalize to 0-1
        
        # Plot the blurred line with dynamic alpha
        ax.plot(x_blur, y_blur, color=color, alpha=alpha.mean(), linewidth=0.5)

# Create a figure with two subplots (side by side)
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

# Set figure background color to white
fig.patch.set_facecolor('white')

# Set background color inside the plot area (axes) to light gray
ax.set_facecolor('white')

# Function to plot points and connect centroids
def plot_centroids(ax,df, coord_x, coord_y, title, xlabel, ylabel,legend=False):
    dffunc=df.copy()
    for i, row in dffunc.iterrows():
        # Extract centroid coordinates for the row
        centroids = [
            row['centroid_1'], 
            row['centroid_2'], 
            row['centroid_3'], 
            row['centroid_4']
        ]
        # a = 0.3 if row['region'] == 'All' else 0.2
        # for j, centroid in enumerate(centroids, start=1):
        #     if j==1:
        #         centroid[coord_x]=0.49*(1-a)+a*centroid[coord_x]
        #     if j==2:
        #         centroid[coord_x]=0.74*(1-a)+a*centroid[coord_x]
        #     if j==3:
        #         centroid[coord_x]=0.52*(1-a)+a*centroid[coord_x]
        #     if j==4:
        #         centroid[coord_x]=0.77*(1-a)+a*centroid[coord_x]               
                
        # Extract coordinates for lines and points
        x_coords = [c[coord_x] for c in centroids] + [centroids[0][coord_x]]  # Close the loop
        y_coords = [c[coord_y] for c in centroids] + [centroids[0][coord_y]]  # Close the loop
        
        # Color based on the region
        line_color = '#D50032' if row['region'] == 'All' else 'darkgray'
        alpha_value = 0.6 if row['region'] == 'All' else 0.01
        alpha_point = 0.99 
        # Draw blurry lines for regions other than "All"
        if row['region'] != 'All':
            plot_blurry_lines(ax, x_coords, y_coords, color=line_color, base_alpha=0.03, blur_iterations=75)
        else:
            ax.plot(x_coords, y_coords, color=line_color, alpha=alpha_value, linewidth=1.5)
        
        # Plot individual centroids
        for j, centroid in enumerate(centroids, start=1):
            x, y = centroid[coord_x], centroid[coord_y]
            color = all_colors[f'centroid_{j}'] if row['region'] == 'All' else 'darkgray'
            size = (row['CHI_score'])**0.5 * 0.8  # Scaling the size for visibility
            ax.scatter(x, y, c=color, s=size, alpha=0.99)
    
    # Set axis labels and title
    ax.set_title(title)
    ax.set_xlabel(xlabel,fontsize=18)
    ax.set_ylabel(ylabel,fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)
    if legend:
        colors = {
            'A': '#4169E1',
            'B': '#50C878',
            'C': '#FFBF00',
            'D': '#D50032'
        }
        legend_labels = ['Cluster 4', 'Cluster 3', 'Cluster 2', 'Cluster 1']  # Desired order
        handles = [plt.Rectangle((0, 0), 1, 1, color=colors[key]) for key in ['D', 'C', 'B', 'A']]  # Match colors with order
    
        ax.legend(handles, legend_labels, loc='lower left', fontsize=18)
    ax.grid(True, color='white', linestyle='-', linewidth=0.5)

# Plot the first scatter plot (X = first value, Y = second value)
# dfchi1=dfchi.copy()
# plot_centroids(
#     ax[0],dfchi, coord_x=2, coord_y=0,
#     title='',
#     xlabel='',
#     ylabel='Normalized PC1'
# )
dfchi2=dfchi.copy()
# Plot the second scatter plot (X = third value, Y = second value)
plot_centroids(
    ax,dfchi, coord_x=2, coord_y=0,
    title='',
    xlabel='Normalized Study Index',
    ylabel='Normalized PC1',
    legend=True
)
dfchi3=dfchi.copy()
# Show the plot
plt.tight_layout()
plt.show()

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
import matplotlib.pyplot as plt
import numpy as np


# Extract unique values from the columns
part1_unique = result['part1'].unique()
part2_unique = result['part2'].unique()
part3_unique = result['part3'].unique()

# Create the plot
fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
ax.set_aspect('equal')
ax.axis('off')

# Function to plot points in a circular layout
def plot_ring(ax, radius, num_points, start_angle=0):
    angles = np.linspace(start_angle, 2 * np.pi + start_angle, num_points, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    points = list(zip(x, y))
    for (i, point) in enumerate(points):
        ax.plot(point[0], point[1], 'o', color='black', markersize=5)  # plot point
    return points

# Plot the inner ring (part3)
inner_ring_points = plot_ring(ax, radius=2, num_points=len(part3_unique))

# Plot the medium ring (part2), arranging the points so there is a point to the left, right, up, down, etc.
medium_ring_points = []
medium_ring_radius = 4
# Calculate angle for the 20 points (we split 360° into 4 quadrants with 5 points each)
angles = np.linspace(0, 2 * np.pi, 20, endpoint=False)
angle_adjustment =0  # Start at 45 degrees to adjust for quadrants

# Adjust for directions (up, down, left, right) and evenly distribute within quadrants
for i in range(20):
    # Adjust angle to give the desired directions
    angle = angles[i-2] + angle_adjustment
    x = medium_ring_radius * np.cos(angle)
    y = medium_ring_radius * np.sin(angle)
    medium_ring_points.append((x, y))

# Plot the medium ring with the adjusted points
for point in medium_ring_points:
    ax.plot(point[0], point[1], 'o', color='black', markersize=5)

# Function to plot branches (lines between points)
def plot_branches(ax, ring1_points, ring2_points, branches_per_point):
    num_ring1_points = len(ring1_points)
    num_ring2_points = len(ring2_points)
    
    # Connect each point in the inner ring to 5 points in the medium ring
    for i in range(num_ring1_points):
        start_point = ring1_points[i]
        
        # The 5 points in the medium ring to connect to
        for j in range(branches_per_point):
            medium_index = (i * branches_per_point + j) % num_ring2_points  # wrap around if necessary
            end_point = ring2_points[medium_index]
            # Draw the line (branch)
            ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], color='black', lw=1)

# Plot branches from inner ring to medium ring
plot_branches(ax, inner_ring_points, medium_ring_points, 5)
# Plot branches from inner ring to medium ring

# Plot the outer ring (80 points)
# Plot the medium ring (part2), arranging the points so there is a point to the left, right, up, down, etc.
outer_ring_points = []
outer_ring_radius=6
# Calculate angle for the 20 points (we split 360° into 4 quadrants with 5 points each)
angles = np.linspace(0, 2 * np.pi, 80, endpoint=False)
angle_adjustment =-2*np.pi/(160)  # Start at 45 degrees to adjust for quadrants

# Adjust for directions (up, down, left, right) and evenly distribute within quadrants
for i in range(80):
    # Adjust angle to give the desired directions
    angle = angles[i-9] + angle_adjustment
    x = outer_ring_radius * np.cos(angle)
    y = outer_ring_radius * np.sin(angle)
    outer_ring_points.append((x, y))

# Function to plot branches from medium ring to outer ring
def plot_outer_branches(ax, ring2_points, ring3_points, branches_per_point):
    num_ring2_points = len(ring2_points)
    num_ring3_points = len(ring3_points)
    
    # Connect each point in the medium ring to 4 points in the outer ring
    for i in range(num_ring2_points):
        start_point = ring2_points[i]
        
        # The 4 points in the outer ring to connect to
        for j in range(branches_per_point):
            outer_index = (i * branches_per_point + j) % num_ring3_points  # wrap around if necessary
            end_point = ring3_points[outer_index]
            # Draw the line (branch)
            ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], color='black', lw=1)

# Plot branches from medium ring to outer ring
plot_outer_branches(ax, medium_ring_points, outer_ring_points, 4)
# Create a DataFrame to store information for the outer ring points
outer_ring_info = []

# Create the mapping from each outer ring point to its corresponding `part1`, `part2`, `part3`
# Each point on the outer ring is connected to 4 points on the medium ring, which in turn are connected to points on the inner ring
id_counter = 1
for i, outer_point in enumerate(outer_ring_points):
    # Find the medium ring points connected to the current outer point
    medium_indices = [(i * 4 + j) % len(medium_ring_points) for j in range(4)]  # 4 branches per medium point
    
    # Get the corresponding part1, part2, part3 for each of the medium ring connections
    part1_values = []
    part2_values = []
    part3_values = []
    
    for medium_index in medium_indices:
        # Find corresponding part1, part2, and part3 values for each connection
        part1 = part1_unique[medium_index % len(part1_unique)]
        part2 = part2_unique[medium_index % len(part2_unique)]
        # Find the inner ring connection (part3)
        inner_index = (medium_index // 5) % len(inner_ring_points)  # 5 branches per inner point
        part3 = part3_unique[inner_index % len(part3_unique)]
        
        part1_values.append(part1)
        part2_values.append(part2)
        part3_values.append(part3)
    
    # Create a row for the outer ring point with its associated part1, part2, part3 values
    outer_ring_info.append({
        'id': id_counter,
        'part1': ', '.join(part1_values),  # Join part1 values if there are multiple
        'part2': ', '.join(part2_values),  # Join part2 values if there are multiple
        'part3': ', '.join(part3_values)   # Join part3 values if there are multiple
    })
    
    id_counter += 1
# Create the DataFrame
outer_ring_df = pd.DataFrame(outer_ring_info)
# Add ID labels to the outer ring
for i, outer_point in enumerate(outer_ring_points):
    x, y = [outerpoint*1.1 for outerpoint in outer_point]
    ax.text(x, y, str(i + 1), color='black', ha='center', va='center', fontweight='bold', fontsize=8)
sorted_result=result.copy()
sorted_result=sorted_result.sort_values(by=['part3', 'part2', 'part1'])
sorted_result = sorted_result.reset_index(drop=True)
colors = {
    "Group1":"#E63946",  # Red
    "Group2":"#F1FAEE",  # Light Gray
    "Group3":"#A8DADC",  # Light Blue
    "Group4":"#457B9D",  # Blue
    "Group5":"#1D3557",  # Dark Blue
    "Group6":"#F1C40F",  # Yellow
    "Group7":"#2ECC71"   # Green
}
# Plot the medium ring with the adjusted points
for i, point in enumerate(outer_ring_points):
    ax.plot(point[0], point[1], 'o', color=colors[sorted_result.iloc[i]["unique_new_groups"]], markersize=5)
    # ax.plot(point[0], point[1], 'o', color="#E63946", markersize=5)
# Display the plot
plt.show()

#%%
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure()
# Calculate the mean of 'riscaldament' grouped by 'new_group'
group_means = DF_groups.groupby('new_group')['riscaldament'].mean().sort_values()

# Reorder the 'new_group' categories based on the sorted means
ordered_groups = group_means.index

# Create the boxplot with the reordered 'new_group'
sns.boxplot(data=DF_groups, x='new_group', y='riscaldament', order=ordered_groups)

# Add labels and title (optional)
plt.xlabel('New Group')
plt.ylabel('Riscaldament')
plt.title('Boxplot of Riscaldament by New Group (Sorted by Mean)')

# Show the plot
plt.show()
plt.figure()
# Filter out the rows where 'new_group_r' is 'Zero'
filtered_df = DF_groups[DF_groups['new_group_r'] != 'Zero']

# Calculate the mean of 'raffrescament' grouped by 'new_group_r'
group_means = filtered_df.groupby('new_group_r')['raffrescament'].mean().sort_values()

# Reorder the 'new_group_r' categories based on the sorted means
ordered_groups = group_means.index

# Create the boxplot with the reordered 'new_group_r'
sns.boxplot(data=filtered_df, x='new_group_r', y='raffrescament', order=ordered_groups)

# Add labels and title (optional)
plt.xlabel('New Group (r)')
plt.ylabel('Raffrescament')
plt.title('Boxplot of Raffrescament by New Group (r), Sorted by Mean')

# Show the plot
plt.show()


#%%
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

# Assuming DF_groups is your dataframe
# Step 1: Select only the relevant columns for the ANOVA
relevant_columns = ['age', 'area', 'degree_days', 'segment', 'abitazione', 'isolamento', 'ripartizione', 'riscaldament']
df_filtered = DF_groups[relevant_columns]

# Step 2: Encode categorical variables as factors
categorical_columns = ['segment', 'abitazione', 'isolamento', 'ripartizione']
df_encoded = df_filtered.copy()

# Convert categorical variables into categorical dtype (factors)
for col in categorical_columns:
    df_encoded[col] = df_encoded[col].astype('category')

# Step 3: Define the formula for the ANOVA model
# We use the "ols" method from statsmodels to fit the model and then apply anova_lm
# For this example, let's fit a model with all independent variables and check their effects
formula = 'riscaldament ~ age + area + degree_days + segment + abitazione + isolamento + ripartizione'

# Fit the ANOVA model
model = smf.ols(formula=formula, data=df_encoded).fit()

# Step 4: Perform ANOVA
anova_results = anova_lm(model)

# Step 5: Print the ANOVA table
print(anova_results)


