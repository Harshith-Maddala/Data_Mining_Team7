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
## SMART QUESTION 3
## Which games and game categories consistently reach the highest peak concurrent users, and does this trend differ significantly across genres?

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

# %%[markdown]
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

# %% [markdown]
#### Distribution of peak ccu across all games

games_df_cleaned['Log_Peak_CCU'] = np.log1p(games_df_cleaned['Peak CCU'])

plt.figure(figsize=(10, 6))
sns.histplot(games_df_cleaned['Log_Peak_CCU'], bins=50, kde=True, color='steelblue', alpha=0.8)
plt.title("Log-Transformed Distribution of Peak CCU", fontsize=16)
plt.xlabel("Log(Peak CCU)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.show()

# %% [markdown]
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

#### Analyzing Peak CCU over time 
# Converting 'Release date' to datetime
games_df_cleaned['Release date'] = pd.to_datetime(games_df_cleaned['Release date'], errors='coerce')

# Extracting the release year
games_df_cleaned['Release Year'] = games_df_cleaned['Release date'].dt.year

# %% [markdown]
#### Average Peak CCU per year
yearly_peak_ccu = games_df_cleaned.groupby('Release Year')['Peak CCU'].mean().reset_index()

plt.figure(figsize=(12, 6))
sns.lineplot(data=yearly_peak_ccu, x='Release Year', y='Peak CCU', marker='o', color='darkcyan')
plt.title('Average Peak CCU by Release Year', fontsize=16)
plt.xlabel('Release Year', fontsize=12)
plt.ylabel('Average Peak CCU', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# %% [markdown]
#### Top Publishers and Developers by Peak CCU

publisher_peak_ccu = games_df_cleaned.groupby('Publishers')['Peak CCU'].sum().reset_index()
top_publishers = publisher_peak_ccu.nlargest(10, 'Peak CCU')

plt.figure(figsize=(12, 6))
sns.barplot(data=top_publishers, x='Peak CCU', y='Publishers', palette='coolwarm')
plt.title('Top 10 Publishers & Developers by Total Peak CCU', fontsize=16)
plt.xlabel('Total Peak CCU', fontsize=12)
plt.ylabel('Publishers & Developers', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# %% [markdown]

#### Multivariate Analysis - Peak CCU vs Price by Categories 

top_categories = games_df_cleaned.groupby('Categories')['Peak CCU'].mean().nlargest(10).index
filtered_df = games_df_cleaned[games_df_cleaned['Categories'].isin(top_categories)]

plt.figure(figsize=(12, 6))
sns.boxplot(data=filtered_df, x='Categories', y='Peak CCU', palette='viridis')
plt.yscale('log')  # Log scale for better visibility
plt.title('Peak CCU Distribution by Top Categories', fontsize=16)
plt.xlabel('Categories', fontsize=12)
plt.ylabel('Peak CCU (Log Scale)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# %% [markdown]

#### Analyzing Peak CCU by Genre and Price
##### To check if the trends in peak concurrent users differ across genres and prices.

#%%
# Calculate average Peak CCU and Price per genre
genre_stats = games_df_cleaned.groupby('Genres').agg({
    'Price': 'mean',
    'Peak CCU': 'mean'
}).sort_values(by='Peak CCU', ascending=False)

genre_stats = genre_stats.reset_index()


# %%

import plotly.express as px
fig = px.scatter(
    genre_stats,
    x='Price',
    y='Peak CCU',
    size='Peak CCU',
    color='Genres',
    hover_name='Genres',
    title='Interactive Price vs Peak CCU by Genres',
    labels={'Price': 'Average Price ($)', 'Peak CCU': 'Average Peak CCU'},
    size_max=50
)

fig.show()

#%% [markdown]
# #### High Peak CCU and Low Price:
#
#### Genres like Massively Multiplayer stand out with the highest average Peak CCU (3000) and a relatively low average price ($2-$3). This highlights the mass appeal of multiplayer games and their potential for attracting large player bases at lower price points.
#
# #### Moderate Price and Peak CCU:
#
#### Genres like RPG, Simulation, and Strategy have moderate Peak CCU (500–1000) and are priced slightly higher (~$4–$6). These genres balance popularity and monetization effectively.
#
# #### Low Peak CCU and Low Price:
#
#### Genres such as Photo Editing and Utilities have lower Peak CCU despite being low-priced (~$1–$2). These may appeal to niche audiences or require better marketing strategies.
#
# #### Premium Genres with Low CCU:
#
#### Genres like Video Production show higher average prices (~$7–$8) but relatively low Peak CCU. This suggests they cater to a specific, possibly professional, audience rather than mass-market appeal.


# %% [markdown]

#### Peak CCU distribution by Top Genres and Categories

# Filter the top genres and categories
top_categories = games_df_cleaned.groupby('Categories')['Peak CCU'].mean().nlargest(5).index
top_genres = games_df_cleaned.groupby('Genres')['Peak CCU'].mean().nlargest(5).index

filtered_df = games_df_cleaned[
    (games_df_cleaned['Categories'].isin(top_categories)) &
    (games_df_cleaned['Genres'].isin(top_genres))
]

# Creating a FacetGrid for Peak CCU distribution by genres and categories
g = sns.catplot(
    data=filtered_df,
    x='Categories',
    y='Peak CCU',
    hue='Genres',
    kind='box',
    height=6,
    aspect=2,
    palette='viridis'
)

g.set(yscale='log')  # Log scale for Peak CCU
g.set_axis_labels('Categories', 'Peak CCU (Log Scale)', fontsize=12)
g.fig.suptitle('Peak CCU Distribution by Categories and Genres', fontsize=16, y=1.02)
g.set_xticklabels(rotation=45, ha='right')
g.add_legend(title='Genres')
plt.tight_layout()
plt.show()

# %% [markdown]

#### STATISTICAL TESTS 
##### Here are some statistical tests to analyze the relationship between Peak CCU, Categories, and Genres.

# %% [markdown]

#### 1. T-Test to compare peak CCU between 2 categories

from scipy.stats import ttest_ind

# Comparing Peak CCU between "Single-player" and "Multi-player"
single_player_ccu = games_df_cleaned[games_df_cleaned['Categories'] == 'Single-player']['Peak CCU']
multi_player_ccu = games_df_cleaned[games_df_cleaned['Categories'] == 'Multi-player']['Peak CCU']

t_stat, p_value = ttest_ind(single_player_ccu, multi_player_ccu, equal_var=False)

print(f"T-Test: Single-player vs Multi-player")
print(f"T-Statistic = {t_stat:.2f}, p-value = {p_value:.4f}")

# %% [markdown]

#### Boxplot to compare the Peak CCU for Single-player and Multiplayer games
plt.figure(figsize=(8, 6))
sns.boxplot(data=games_df_cleaned[games_df_cleaned['Categories'].isin(['Single-player', 'Multi-player'])],
            x='Categories', y='Peak CCU', palette='coolwarm')
plt.yscale('log')  # Log scale for better visualization
plt.title('Comparison of Peak CCU: Single-player vs Multi-player', fontsize=14)
plt.xlabel('Categories', fontsize=12)
plt.ylabel('Peak CCU (Log Scale)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# %% [markdown]

#### Investigating the specific games contributing to the outliers in both categories.
outliers = games_df_cleaned[
    (games_df_cleaned['Peak CCU'] > 1e5) & 
    (games_df_cleaned['Categories'].isin(['Single-player', 'Multi-player']))
]
print(outliers[['Name', 'Categories', 'Peak CCU']])

# %% [markdown]

#### 2. ANOVA test for categories

from scipy.stats import f_oneway
anova_categories = f_oneway(
    *[group['Peak CCU'].values for _, group in games_df_cleaned.groupby('Categories')]
)

print(f"ANOVA for Categories: F-Statistic = {anova_categories.statistic:.2f}, p-value = {anova_categories.pvalue:.4f}")


# %% [markdown]

#### 3. ANOVA Test for Genres

anova_genres = f_oneway(
    *[group['Peak CCU'].values for _, group in games_df_cleaned.groupby('Genres')]
)

print(f"ANOVA for Genres: F-Statistic = {anova_genres.statistic:.2f}, p-value = {anova_genres.pvalue:.4f}")

# %% [markdown]

#### 4. Tukey's HSD for categories - which specific categories differ significantly? 

from statsmodels.stats.multicomp import pairwise_tukeyhsd

tukey_categories = pairwise_tukeyhsd(
    games_df_cleaned['Peak CCU'], 
    games_df_cleaned['Categories']
)
print(tukey_categories.summary())

# %% [markdown]

#### 5. Tukey's HSD for Genres - which specific genres differ significantly?

tukey_genres = pairwise_tukeyhsd(
    games_df_cleaned['Peak CCU'], 
    games_df_cleaned['Genres']
)
print(tukey_genres.summary())

# %% [markdown]

#### Let's predict the Peak CCU by using features like categories, genres, price and others. I feel XGBoost or LightGBM are ideal in this case. 

#### 1. Feature Engineering

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

# Categorical features
categorical_features = ['Categories', 'Genres']

# Numerical features
numerical_features = ['Price', 'Recommendations', 'Achievements']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numerical_features)
    ]
)

# Target variable: log-transform for stability
games_df_cleaned['Log_Peak_CCU'] = np.log1p(games_df_cleaned['Peak CCU'])
y = games_df_cleaned['Log_Peak_CCU']

# Define X (features)
X = games_df_cleaned[categorical_features + numerical_features]

# %% [markdown]

#### 2. Test-train split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% [markdown]

#### 3. XGBoost Regression




