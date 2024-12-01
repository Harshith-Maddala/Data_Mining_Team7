#%%[markdown]
#
# # Beyond the Play Button : Insights and Trends from Steam’s Game Library
# ## By: Neeraj Shashikant Magadum, Aditya Kanbargi, Harshith Maddala, Sanjana Muralidhar
# ### Date: 17 Nov
#
# 

#%%[markdown]
# ## 1. Importing Required Libraries

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import rfit 
import plotly.express as px


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

# %% [markdown]
# ## 6. Final Check for Missing Values After Cleaning

# After cleaning, check if there are any remaining missing values in the cleaned dataset.
print(games_df_cleaned.isnull().sum())
games_df_cleaned.shape

games_df_cleaned.to_csv("games_df_cleaned.csv")

games_df_cleaned.head()


# %% [markdown] 

## Which games and game categories (e.g., single-player, multiplayer) consistently reach the highest peak concurrent users, and does this trend differ significantly across genres and game prices?
## Interchanging column names 
col_names = list(games_df_cleaned.columns)
print(col_names)
col_names[-1], col_names[-2] = col_names[-2], col_names[-1]
games_df_cleaned.columns = col_names

games_df_cleaned.head()







# %% [markdown] 
## SMART QUESTION 1
#
## What is the average price of games within each genre on Steam, and which specific genres have the highest and lowest average prices? 

# %%
games_df_cleaned['Genres']

# %% [markdown]

# ### Step 1: Exploding Genres for Analysis: Data preperation
# A game can belong to multiple genres; we will create one row per genre for accurate analysis.

# %%
games_df_cleaned['Genres'] = games_df_cleaned['Genres'].str.split(',')
games_df_cleaned['Genres'] = games_df_cleaned['Genres'].apply(lambda x: [g.strip().title() for g in x])

exploded_df = games_df_cleaned.explode('Genres')

# removing outliers
exploded_df = exploded_df[exploded_df['Price'] < exploded_df['Price'].quantile(0.99)]


# %% 
# Identifying unique genres
exploded_df.head()
exploded_df["Genres"].unique()

# %% [markdown]
# ### Step 2:  Identifying Top 10 genres 

# %%
all_g = []
genres =exploded_df["Genres"].dropna().values
for g in genres:
    for word in g.split(",") :
        all_g.append(word)


genres_not_unique = pd.DataFrame(all_g , columns=["genres"])
genres_unique_counts = genres_not_unique.groupby(['genres'])['genres'].count()
genres_unique_counts = genres_unique_counts.sort_values(ascending=False)


genres_unique_counts.head(10)

# %%
top_10_genres = list(genres_unique_counts[:10].index)
top_10_genres

# %%
import plotly.express as px
genres_data = genres_unique_counts[:10].sort_values(ascending=False)

fig = px.bar(genres_data,x=genres_data.values,y=genres_data.index,orientation='h',labels={'x': 'Count', 'index':'Genre'},color=genres_data.index,
            title="Number of Games in the Top 10 Genres")

fig.show()


# %% [markdown]
# ### Step 3: Calculating Average Price Per Genre- Descriptive statistics

# %%
# Calculate the average price for each genre
avg_price_per_genre = exploded_df.groupby('Genres')['Price'].mean().sort_values(ascending=False)

# Identify the genres with the highest and lowest average prices
highest_avg_price_genre = avg_price_per_genre.idxmax()
lowest_avg_price_genre = avg_price_per_genre.idxmin()

highest_avg_price = avg_price_per_genre.max()
lowest_avg_price = avg_price_per_genre.min()

# Display Results
print(f"Genre with the Highest Average Price: {highest_avg_price_genre} (${highest_avg_price:.2f})")
print(f"Genre with the Lowest Average Price: {lowest_avg_price_genre} (${lowest_avg_price:.2f})")

# Display the full list of average prices for all genres
print("\nAverage Price per Genre:")
print(avg_price_per_genre)


# %% [markdown]
# ### Step 4: Visualizing Average Price Per Genre

# %%


# Top 10 Genres with Highest Average Prices
plt.figure(figsize=(10, 6))
avg_price_per_genre.head(10).plot(kind='bar', color='green')
plt.title('Top 10 Genres with Highest Average Prices', fontsize=14)
plt.xlabel('Genre', fontsize=12)
plt.ylabel('Average Price ($)', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Top 10 Genres with Lowest Average Prices
plt.figure(figsize=(10, 6))
avg_price_per_genre.tail(10).plot(kind='bar', color='orange')
plt.title('Top 10 Genres with Lowest Average Prices', fontsize=14)
plt.xlabel('Genre', fontsize=12)
plt.ylabel('Average Price ($)', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%% [markdown]
# ### Step 5: Compare average prices across genres using ANOVA.
#
# ### Defining our null and alternate Hypothesis
#
# #### Null Hypothesis H0 : The average price of games is the same across all genres.
# #### Alternate Hypothesis H1 : The average price of games differs across at least one genre

#%%
from scipy.stats import f_oneway

# Group by genres
genre_groups = [group['Price'].dropna() for _, group in exploded_df.groupby('Genres')]

# Perform ANOVA
anova_result = f_oneway(*genre_groups)
print(f"ANOVA Result: F-statistic={anova_result.statistic:.2f}, p-value={anova_result.pvalue:.2e}")

#%% [markdown]
# ##### The very small p-value (less than the standard threshold of 0.05) indicates strong evidence to reject the null hypothesis.
#
# ##### A higher F-statistic indicates a greater variance between groups compared to the variance within groups.
#
# ### There is strong statistical evidence that the average price of games is not the same across all genres. 
# ##### This means at least one genre has a significantly different average price compared to others.
#
# ##### To determine which genres differ, let us perform a post hoc test - Tukey’s HSD.

# %%
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Perform Tukey's HSD test
tukey = pairwise_tukeyhsd(endog=exploded_df['Price'], groups=exploded_df['Genres'], alpha=0.05)
print(tukey)


# %% [markdown]
# ### Key Findings:
#
# 1.	Significant Differences:
# Certain genre pairs exhibit statistically significant differences in their average prices (Reject=True), indicating that their price means differ beyond random variation.
# Example: Video Production vs. Violent: Mean Difference = -11.96.
# Interpretation: Violent games are significantly cheaper than Video Production games.
#
# 2.	Non-Significant Differences:
# Other genre pairs show no statistically significant difference (Reject=False), meaning the average prices between those genres are similar.
# Example: Video Production vs. Web Publishing: Mean Difference = -1.063.
# Interpretation: The average prices for these genres are not significantly different.

# %%

tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])

# Filter for significant pairs
significant_pairs = tukey_df[tukey_df['reject'] == True]

# Pivot for heatmap
significant_matrix = significant_pairs.pivot(index='group1', columns='group2', values='meandiff')


plt.figure(figsize=(12, 8))
sns.heatmap(significant_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Significant Mean Differences Between Groups')
plt.show()

# %% [markdown]

# #### 1. Key Genres with Higher Prices (Red Clusters):
# 
# Genres like Web Publishing, Video Production, and Simulation often feature higher prices, as evidenced by the positive mean differences when compared to cheaper genres.
#
# #### 2. Key Genres with Lower Prices (Blue Clusters):
#
# Genres like Free To Play, Violent, and Casual consistently have lower average prices, as seen in their negative mean differences with higher-priced genres.
#
# #### 3. Widest Price Gaps:
#
# The largest price gaps occur between genres like Video Production and Violent or Web Publishing and Free To Play.
#
# Some genre pairs (lighter colors) show mean differences close to zero, indicating that their average prices are not significantly different (e.g., Simulation and Sports).

# %% [markdown]
# ### Step 6: MODEL BUILDING 




