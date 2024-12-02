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

#%%[markdown]
# Interchanging columns and data prep 

# %%
col_names = list(games_df_cleaned.columns)
print(col_names)
col_names[-1], col_names[-2] = col_names[-2], col_names[-1]
games_df_cleaned.columns = col_names

games_df_cleaned.head()

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
## SMART QUESTION 1
#
##### What is the average price of games within each genre and across categories on Steam, and which specific genres or categories have the highest and lowest average prices? 

# %%
games_df_cleaned['Genres']

# %% [markdown]

# ### Step 1: Exploding Genres and categories for Analysis: Data preperation
# A game can belong to multiple genres; we will create one row per genre for accurate analysis.
# A game can belong to multiple categories; we will create one row per category for accurate analysis.

# %%
#### There are multiple values in 'Categories' and 'Genres' columns, let's split and explode them into multiple rows. 

games_df_cleaned['Categories'] = games_df_cleaned['Categories'].str.split(',')
games_df_cleaned['Genres'] = games_df_cleaned['Genres'].str.split(',')

games_df_cleaned = games_df_cleaned.explode('Categories').explode('Genres').reset_index(drop=True)
games_df_cleaned.head()

# removing outliers
games_df_cleaned = games_df_cleaned[games_df_cleaned['Price'] < games_df_cleaned['Price'].quantile(0.99)]

# %% 
# Identifying unique genres
games_df_cleaned["Genres"].unique()

#%%
# Identifying unique categories
games_df_cleaned["Categories"].unique()

# %% [markdown]
# ### Step 2:  Identifying Top 10 genres and categories

# %%
import plotly.express as px
all_g = []
genres =games_df_cleaned["Genres"].dropna().values
for g in genres:
    for word in g.split(",") :
        all_g.append(word)


genres_not_unique = pd.DataFrame(all_g , columns=["genres"])
genres_unique_counts = genres_not_unique.groupby(['genres'])['genres'].count()
genres_unique_counts = genres_unique_counts.sort_values(ascending=False)


print(genres_unique_counts.head(10))


top_10_genres = list(genres_unique_counts[:10].index)
top_10_genres


genres_data = genres_unique_counts[:10].sort_values(ascending=False)


fig = px.bar(
    genres_data,
    x=genres_data.values,
    y=genres_data.index,
    orientation='h',
    labels={'x': 'Count', 'index': 'Category'},
    color=genres_data.index,
    title="Number of Games in the Top 10 Genres",
    width=800  # Adjust the width here
)

fig.show()


#%%
all_c = []
categories = games_df_cleaned["Categories"].dropna().values
for c in categories:
    for word in c.split(","):
        all_c.append(word.strip())

# Create a DataFrame for category counts
categories_not_unique = pd.DataFrame(all_c, columns=["categories"])
categories_unique_counts = categories_not_unique.groupby(['categories'])['categories'].count()
categories_unique_counts = categories_unique_counts.sort_values(ascending=False)

# Display the top 10 categories
categories_unique_counts.head(10)


# Extract the top 10 categories
top_10_categories = list(categories_unique_counts[:10].index)
print("Top 10 Categories:", top_10_categories)

# Visualize the top 10 categories using Plotly
categories_data = categories_unique_counts[:10].sort_values(ascending=False)


fig = px.bar(
    categories_data,
    x=categories_data.values,
    y=categories_data.index,
    orientation='h',
    labels={'x': 'Count', 'index': 'Category'},
    color=categories_data.index,
    title="Number of Games in the Top 10 Categories",
    width=800  # Adjust the width here
)

fig.show()


# %% [markdown]
# ### Step 3: Calculating Average Price Per Genre- Descriptive statistics

# %%
# Calculate the average price for each genre
avg_price_per_genre = games_df_cleaned.groupby('Genres')['Price'].mean().sort_values(ascending=False)

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
# ### Step 4: Calculating Average Price for different game categories- Descriptive statistics

# Calculate the average price for each game category
avg_price_per_category = games_df_cleaned.groupby('Categories')['Price'].mean().sort_values(ascending=False)

# Identify the categories with the highest and lowest average prices
highest_avg_price_category = avg_price_per_category.idxmax()
lowest_avg_price_category = avg_price_per_category.idxmin()

highest_avg_price_cat = avg_price_per_category.max()
lowest_avg_price_cat = avg_price_per_category.min()

# Display Results
print(f"Category with the Highest Average Price: {highest_avg_price_category} (${highest_avg_price_cat:.2f})")
print(f"Category with the Lowest Average Price: {lowest_avg_price_category} (${lowest_avg_price_cat:.2f})")

# Display the full list of average prices for all categories
print("\nAverage Price per Category:")
print(avg_price_per_category)




# %% [markdown]
# ### Step 5: Visualizing the top genres and categories with highest average price

# %%

# Create a figure with subplots and adjust spacing
fig, axes = plt.subplots(2, 2, figsize=(20, 12))  # Increase the figure size for better visibility

# Plot for Top 10 Genres with Highest Average Prices
avg_price_per_genre.head(10).plot(
    kind='bar', ax=axes[0, 0], color='green', title="Top 10 Genres with Highest Average Prices"
)
axes[0, 0].set_xlabel('Genre', fontsize=12)
axes[0, 0].set_ylabel('Average Price ($)', fontsize=12)
axes[0, 0].tick_params(axis='x', rotation=45, labelsize=10)

# Plot for Top 10 Genres with Lowest Average Prices
avg_price_per_genre.tail(10).plot(
    kind='bar', ax=axes[0, 1], color='orange', title="Top 10 Genres with Lowest Average Prices"
)
axes[0, 1].set_xlabel('Genre', fontsize=12)
axes[0, 1].set_ylabel('Average Price ($)', fontsize=12)
axes[0, 1].tick_params(axis='x', rotation=45, labelsize=10)

# Plot for Top 10 Categories with Highest Average Prices
avg_price_per_category.head(10).plot(
    kind='bar', ax=axes[1, 0], color='blue', title="Top 10 Categories with Highest Average Prices"
)
axes[1, 0].set_xlabel('Category', fontsize=12)
axes[1, 0].set_ylabel('Average Price ($)', fontsize=12)
axes[1, 0].tick_params(axis='x', rotation=45, labelsize=10)

# Plot for Top 10 Categories with Lowest Average Prices
avg_price_per_category.tail(10).plot(
    kind='bar', ax=axes[1, 1], color='red', title="Top 10 Categories with Lowest Average Prices"
)
axes[1, 1].set_xlabel('Category', fontsize=12)
axes[1, 1].set_ylabel('Average Price ($)', fontsize=12)
axes[1, 1].tick_params(axis='x', rotation=45, labelsize=10)

# Add a common title for all subplots
fig.suptitle('Analysis of Average Game Prices Across Genres and Categories', fontsize=20, fontweight='bold')

# Adjust layout to ensure titles and labels don't overlap
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the rect parameter for the title
plt.show()



#%%
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Create a pivot table with Genres as rows and Categories as columns
pivot_table = games_df_cleaned.pivot_table(
    values='Price', 
    index='Genres', 
    columns='Categories', 
    aggfunc='mean'
)


# Step 2: Fill NaN values with 0 or another placeholder to represent missing data
pivot_table.fillna(0, inplace=True)

# Step 3: Plot the heat map
plt.figure(figsize=(15, 10))  # Adjust the size as needed
sns.heatmap(
    pivot_table, 
    annot=False,  # Set to True to see the numerical values
    cmap='coolwarm',  # Choose a color map
    cbar_kws={'label': 'Average Price ($)'}  # Add a color bar with a label
)
plt.title('Average Price Across Different Genres and Categories', fontsize=16)
plt.xlabel('Categories', fontsize=12)
plt.ylabel('Genres', fontsize=12)
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
plt.tight_layout()
plt.show()

#%% [markdown]
# High-Priced Genres and Categories:
#
# Genres like "Simulation" and "Audio Production" command the highest prices, likely due to their niche appeal and specialized features.
# Categories such as "SteamVR Collectibles" and "Remote Play on TV" align with premium experiences, targeting advanced gaming setups.
#
# Low-Priced Genres and Categories:
#
# Genres like "Free to Play" and "Massively Multiplayer" are among the lowest-priced, relying on wider audience appeal and alternative revenue models like in-app purchases.
# Categories such as "Mods" and "In-App Purchases" are also associated with lower-priced games, supporting broader accessibility.
#
# Granular Insights from the Heatmap:
#
# Genre-Category Intersections: Specific combinations like "Video Production" with VR categories show the highest average prices, indicating a premium segment for innovation-heavy experiences.
#
# Broad Trends: Most genre-category combinations have moderate to low prices, reflecting a market-wide emphasis on affordability.
#
# Market Strategy Takeaways:
#
# For Developers: Align pricing strategies with the niche or mass-market appeal of the genre or category.
# 
# For Consumers: Premium pricing correlates with advanced features, while affordability dominates in popular categories.


#%% [markdown]
# ### Step 6: Compare average prices across genres using ANOVA.
#
# ### Defining our null and alternate Hypothesis
#
# #### Null Hypothesis H0 : The average price of games is the same across all genres.
# #### Alternate Hypothesis H1 : The average price of games differs across at least one genre

#%%
from scipy.stats import f_oneway

# Group by genres
genre_groups = [group['Price'].dropna() for _, group in games_df_cleaned.groupby('Genres')]

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
tukey = pairwise_tukeyhsd(endog=games_df_cleaned['Price'], groups=games_df_cleaned['Genres'], alpha=0.05)
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

#%% [markdown]
# #### Step 7: CHI SQUARE TEST 
#
# #### Chi-Square Test for Association Between Game Categories and Price Categories

#%%
from scipy.stats import chi2_contingency

# Target: Define clusters or price categories
games_df_cleaned['Price Category'] = pd.qcut(games_df_cleaned['Price'], q=3, labels=['Low', 'Medium', 'High'])
y = games_df_cleaned['Price Category']

# Create a contingency table
contingency_table = pd.crosstab(games_df_cleaned['Categories'], games_df_cleaned['Price Category'])

# Perform Chi-Square Test
chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

print(f"Chi-Square Statistic: {chi2_stat:.2f}, p-value: {p_value:.2e}")


# Plot the heatmap
plt.figure(figsize=(15, 10))
sns.heatmap(
    contingency_table,
    annot=True,
    fmt='d',
    cmap='coolwarm',
    cbar_kws={'label': 'Count'}
)
plt.title('Heatmap of Game Categories vs Price Categories', fontsize=16)
plt.xlabel('Price Category', fontsize=12)
plt.ylabel('Game Categories', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

#%% [markdown]
# Chi-Square Statistic: 48145.96
#
# p-value: 0.00
#
# Since the p-value is less than the standard significance threshold of 0.05, we reject the null hypothesis. This indicates a strong statistical association between game categories and price categories, meaning that the price of a game is significantly influenced by its category. 
#
# Interpretation from heatmap
#
# Category Distribution Across Price Tiers:
#
# The "Single-player" category stands out with a significant concentration in all three price categories, especially in the "Low" price tier, indicating its popularity and affordability.
# Categories like "Multi-player" and "Full controller support" also have notable counts across price categories, with slightly higher representation in the "Medium" and "High" price tiers.
#
# High-Price Categories:
#
# Niche categories such as "VR Only," "SteamVR Collectibles," and "Tracked Motion Controller Support" show higher counts in the "High" price category, likely reflecting the premium nature of these technologies.
# 
# Low-Price Categories:
#
# Common and accessible categories like "Steam Achievements" and "Stats" are heavily concentrated in the "Low" price tier, suggesting they are included in more budget-friendly games.
# 
# Balanced Categories:
#
# Categories like "Shared/Split Screen Co-op" and "Steam Cloud" show relatively balanced distributions across all price tiers, indicating their versatility and inclusion in games with varied pricing strategies.


# %% [markdown]
# ### Step 8: Advanced Insights and Modeling for Average Price Analysis by Genre.
#
# #### Understanding of genre-based pricing trends through clustering and classification model.
