#%%[markdown]
#
# # Beyond the Play Button : Insights and Trends from Steam’s Game Library
# ## By: Neeraj Shashikant Magadum, Aditya Kanbargi, Harshith Maddala, Sanjana Muralidhar
# ### Date: 17 Nov
#
# 

#%%[markdown]
# ## 1. Importing Required Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import rfit 


#%% [markdown]
# ## 2. Loading the Dataset
games_df = pd.read_csv('games.csv', index_col=False)

# Display the first 5 rows of the dataset to get an overview of its structure and content
print(games_df.head())


# %% [markdown]
# ## 3. Basic Exploration of the Dataset

# Check the shape of the dataset (number of rows and columns)
games_df.shape

# Check the data types of each column to understand what kind of data we are working with
games_df.dtypes

# Check for missing values in each column; this helps us identify if any data cleaning is needed
print(games_df.isnull().sum())

# %% [markdown]
# ## 4. Descriptive Statistics for Numerical Columns

# Get summary statistics for numerical columns in the dataset (e.g., mean, min, max, etc.)
print(games_df.describe())


# %% [markdown]
# ## 5. Data Cleaning: Dropping Missing Values and Normalizing Genres
# Define the columns to remove from the dataset
removed_columns = ['AppID', 'Screenshots', 'Reviews', 'Header image', 'Website', 'Support url', 
                   'Support email', 'Metacritic url', 'Notes', 'Average playtime two weeks', 
                   'Median playtime two weeks', 'Median playtime forever', 'Movies',
                   'Score rank']

# Drop these columns from the dataset
games_df_cleaned = games_df.drop(columns=removed_columns).copy()

# # Print the "Genres" column to inspect its content before further processing.
print(games_df_cleaned["Tags"])

# # Convert all genre names to lowercase for consistency (this helps in grouping similar genres later).
games_df_cleaned["Tags"] = games_df_cleaned["Tags"].str.lower()

games_df_cleaned = games_df_cleaned.dropna(subset=['Categories'])
games_df_cleaned = games_df_cleaned.dropna(subset=['Genres'])
games_df_cleaned = games_df_cleaned.dropna(subset=['Tags'])

games_df_cleaned['Developers'].fillna('Unknown', inplace=True)
games_df_cleaned['Publishers'].fillna('Unknown', inplace=True)

## Interchanging column names for appropriateness
col_names = list(games_df_cleaned.columns)
print(col_names)
col_names[-1], col_names[-2] = col_names[-2], col_names[-1]
games_df_cleaned.columns = col_names

# %% [markdown]
# ## 6. Final Check for Missing Values After Cleaning

# After cleaning, check if there are any remaining missing values in the cleaned dataset.
print(games_df_cleaned.isnull().sum())
games_df_cleaned.shape

games_df_cleaned.to_csv("games_df_cleaned.csv")
# %% [markdown] 
## Which games and game categories (e.g., single-player, multiplayer) consistently reach the highest peak concurrent users, and does this trend differ significantly across genres and game prices?

# Rearranging some columns
# Moving data from 'Publishers' to 'Developers'
games_df_cleaned['Developers'] = games_df_cleaned['Publishers']

# Moving data from 'Categories' to 'Publishers'
games_df_cleaned['Publishers'] = games_df_cleaned['Categories']

# Moving data from 'Tags' to 'Categories' and then dropping 'Tags'
games_df_cleaned['Categories'] = games_df_cleaned['Tags']
games_df_cleaned.drop(columns=['Tags'], inplace=True)

# updated DataFrame
games_df_cleaned.head()
# %% [markdown]

### Spliiting and Exploding columns
#
#### There are multiple values in 'Categories' and 'Genres' columns, let's split and explode them into multiple rows. 

games_df_cleaned['Categories'] = games_df_cleaned['Categories'].str.split(',')
games_df_cleaned['Genres'] = games_df_cleaned['Genres'].str.split(',')

df_exploded = games_df_cleaned.explode('Categories').explode('Genres').reset_index(drop=True)
print(df_exploded)

df_exploded.head()
# %% [markdown]

### Basic statistics of Numerical Columns
print("Basic Statistics of Numerical Columns:")
print(games_df_cleaned.describe())


# %% [markdown]

### Let's see the distribution of Peak CCU
# Log Transformation (to reduce skewness)
games_df_cleaned['Log_Peak_CCU'] = np.log1p(games_df_cleaned['Peak CCU'])

plt.figure(figsize=(12, 6))
upper_limit = np.percentile(games_df_cleaned['Log_Peak_CCU'], 95)

sns.histplot(games_df_cleaned['Log_Peak_CCU'], kde=True, bins=50, color='steelblue', alpha=0.8)
plt.title("Log-Transformed Distribution of Peak Concurrent Users (CCU)", fontsize=16)
plt.xlabel("Log(Peak CCU)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.xlim(0, upper_limit)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()


### Categories distribution

# Counting the occurrences of each category
category_counts = df_exploded['Categories'].value_counts()

# Plotting the pie chart
# Grouping smaller categories into 'Other'
threshold = 0.03 * category_counts.sum()
filtered_categories = category_counts[category_counts > threshold]
other_sum = category_counts[category_counts <= threshold].sum()

# Adding 'Other' to the filtered categories
filtered_categories['Other'] = other_sum

plt.figure(figsize=(8, 8))
plt.pie(
    filtered_categories,
    labels=filtered_categories.index,
    autopct='%1.1f%%',
    startangle=90,
    colors=sns.color_palette('pastel'),
    wedgeprops={'edgecolor': 'black'}
)
plt.title("Distribution of Categories", fontsize=16)
plt.show()

