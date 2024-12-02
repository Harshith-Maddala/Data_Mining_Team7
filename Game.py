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

### Splitting and Exploding columns
#
#### There are multiple values in 'Categories' and 'Genres' columns, let's split and explode them into multiple rows. 
games_df_cleaned['Categories'] = games_df_cleaned['Categories'].str.split(',')
games_df_cleaned['Genres'] = games_df_cleaned['Genres'].str.split(',')

games_df_cleaned = games_df_cleaned.explode('Categories').explode('Genres').reset_index(drop=True)
games_df_cleaned.head()


# %% [markdown]

### Basic statistics of Numerical Columns
print("Basic Statistics of Numerical Columns:")
print(games_df_cleaned.describe())


# %% [markdown]

### Let's see the distribution of Peak CCU
### Aggregate stats for peak ccu by categories
category_peak_ccu_stats = games_df_cleaned.groupby('Categories')['Peak CCU'].agg(['mean', 'median', 'max', 'sum', 'count']).reset_index()
category_peak_ccu_stats = category_peak_ccu_stats.sort_values(by='mean', ascending=False)
print(category_peak_ccu_stats.head(10))

# Aggregate statistics for Peak CCU by Genres
genre_peak_ccu_stats = games_df_cleaned.groupby('Genres')['Peak CCU'].agg(['mean', 'median', 'max', 'sum', 'count']).reset_index()
genre_peak_ccu_stats = genre_peak_ccu_stats.sort_values(by='mean', ascending=False)
print(genre_peak_ccu_stats.head(10))

# %% [markdown]

#### Top Categories by Average peak CCU

top_categories = category_peak_ccu_stats.nlargest(10, 'mean')

# Plotting
plt.figure(figsize=(12, 6))
sns.barplot(data=top_categories, x='Categories', y='mean', palette='viridis')
plt.title('Top 10 Categories by Average Peak CCU', fontsize=16)
plt.xlabel('Categories', fontsize=12)
plt.ylabel('Average Peak CCU', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


#### Top Genres by Average Peak CCU

top_genres = genre_peak_ccu_stats.nlargest(10, 'mean')

# Plotting
plt.figure(figsize=(12, 6))
sns.barplot(data=top_genres, x='Genres', y='mean', palette='magma')
plt.title('Top 10 Genres by Average Peak CCU', fontsize=16)
plt.xlabel('Genres', fontsize=12)
plt.ylabel('Average Peak CCU', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#### Distribution of peak ccu across all games

games_df_cleaned['Log_Peak_CCU'] = np.log1p(games_df_cleaned['Peak CCU'])

plt.figure(figsize=(10, 6))
sns.histplot(games_df_cleaned['Log_Peak_CCU'], bins=50, kde=True, color='steelblue', alpha=0.8)
plt.title("Log-Transformed Distribution of Peak CCU", fontsize=16)
plt.xlabel("Log(Peak CCU)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.show()

#### Top 10 games by Peak CCU


top_10_games = games_df_cleaned.groupby('Name').agg({'Peak CCU': 'max'}).nlargest(10, 'Peak CCU').reset_index()
top_10_games = top_10_games.merge(games_df_cleaned[['Name', 'Release date']], on='Name', how='left').drop_duplicates()

# Plotting
plt.figure(figsize=(12, 8))
sns.barplot(data=top_10_games, y='Name', x='Peak CCU', palette='cubehelix')
plt.title('Top 10 Games by Peak CCU', fontsize=16)
plt.xlabel('Peak CCU', fontsize=12)
plt.ylabel('Game Name', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

