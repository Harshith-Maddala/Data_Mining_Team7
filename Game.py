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


games_df_cleaned.head()


# %% [markdown]
### Distribution of Required age from 1 to 21 years
# %% [markdown]
# ## 7. Basic Plots to Understand Few variables in Data

#### Distribution of Required age from 1 to 21 years

# Filter the age above '0' years
filtered_df = games_df_cleaned[games_df_cleaned['Required age'] > 0]

plt.figure(figsize=(10, 6))
sns.countplot(x='Required age', data=filtered_df, palette='viridis', order=sorted(filtered_df['Required age'].unique()))

#plot
plt.title('Distribution of Required Age Ratings (Excluding Age 0)', fontsize=16)
plt.xlabel('Required Age', fontsize=14)
plt.ylabel('Number of Games', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)  
sns.despine() 
plt.show()

# %% [markdown]
#### Distribution of Game Prices
plt.figure(figsize=(12, 7))
sns.histplot(games_df_cleaned['Price'], bins=50, color='dodgerblue', edgecolor='black')

# plot
plt.title('Distribution of Game Prices', fontsize=18)
plt.xlabel('Price ($)', fontsize=14)
plt.ylabel('Number of Games', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)  
sns.despine()  
plt.show()

# %% [markdown]

#### Top 10 Costliest Games

# Sorting
top_10_costliest_games = games_df_cleaned.sort_values(by='Price', ascending=False).head(10)

plt.figure(figsize=(12, 7))
sns.barplot(x='Price', y='Name', data=top_10_costliest_games, palette='viridis', edgecolor='black')

# plot
plt.title('Top 10 Costliest Games on Steam', fontsize=18)
plt.xlabel('Price ($)', fontsize=14)
plt.ylabel('Game Names', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)  
sns.despine()  
plt.show()


# %% [markdown]

#### Top 10 Most Supported Audio Languages

# Splitting the Aduio Languages list to individual Languages
audio_languages_list = games_df_cleaned['Full audio languages'].dropna().str.split(',').sum()
audio_languages_cleaned = [lang.strip().strip("[]'\"") for lang in audio_languages_list]

# Count the occurrences 
language_counts = pd.Series(audio_languages_cleaned).value_counts()

plt.figure(figsize=(12, 7))
sns.barplot(x=language_counts.head(10).values, y=language_counts.head(10).index, palette='coolwarm', edgecolor='black')

# plot 
plt.title('Top 10 Full Audio Languages in Games', fontsize=18)
plt.xlabel('Number of Games', fontsize=14)
plt.ylabel('Languages', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
sns.despine()  
plt.show()

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
##### What pricing patterns emerge within game genres and categories on Steam, and which specific genres or categories exhibit the highest and lowest average prices? How do these trends correlate with game features and audience appeal?

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

# Create a DataFrame for Genre counts
genres_not_unique = pd.DataFrame(all_g , columns=["genres"])
genres_unique_counts = genres_not_unique.groupby(['genres'])['genres'].count()
genres_unique_counts = genres_unique_counts.sort_values(ascending=False)

# Display the top 10 Genres
print(genres_unique_counts.head(10))


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
print(categories_unique_counts.head(10))


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
# ### Step 3:  Calculating Average Price Per Genre- Descriptive statistics

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
# ### Step 5: Visualizing  genres and categories with highest and lowest average price

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
# ### Step 8: Advanced Insights and Modeling for Average Price Analysis by Genre and peak CCU.
#
# #### Understanding of genre-based pricing trends through clustering and classification techniques.

#%% [markdown]
# ### 8.1 : K-Means Clustering: Grouping Genres Based on Similarity 

#%% [markdown]
# The genres have been grouped into 4 clusters based on their average price and average Peak CCU (popularity):
#
# ## Cluster 0 (Budget-Friendly, Low Popularity):
#
# Average Price: $1.25
# Average Peak CCU: 125.46
#
# Genres:
# "360 Video," "Documentary," "Free to Play," "Movie," "Short," "Tutorial," etc.
#
# Insights: This cluster represents genres with low price points and low concurrent users. These are likely to be casual, accessible, or niche genres that prioritize affordability and reach.
#
#
# ## Cluster 1 (Affordable, Moderate Popularity):
# Average Price: $6.10
# Average Peak CCU: 291.49
# 
# Genres:
# "Casual," "Indie," "Utilities," "Game Development," "Massively Multiplayer," etc.
# 
# Insights: This cluster contains mid-range genres that balance affordability with moderate popularity. These genres might cater to a broad audience but don't achieve peak popularity.
# 
# ## Cluster 2 (Premium, High Popularity):
#
# Average Price: $8.80
# Average Peak CCU: 420.04
# 
# Genres:
# "Action," "Adventure," "Simulation," "RPG," "Strategy," etc.
# 
# Insights: This cluster is characterized by higher prices and significant popularity. It includes well-established genres known for immersive or competitive gaming experiences, appealing to dedicated players.
# 
# ## Cluster 3 (High Popularity Outliers):
#
# Average Price: $5.83
# Average Peak CCU: 6702.75
# Genres:
# "Photo Editing"
#
# Insights: This cluster is an outlier due to its exceptionally high Peak CCU, suggesting games in this genre are extremely popular despite their moderate pricing. 
#
# These clusters help uncover patterns like:
#
# Which genres are budget-friendly?
#
# Which genres are highly popular and justify premium pricing?
#
# Outlier genres with unique pricing or popularity trends.
# %%

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Extract relevant features: Aggregate by genre and calculate mean of Price and Peak CCU
cluster_data = games_df_cleaned.groupby('Genres')[['Price', 'Peak CCU']].mean().reset_index()

# Drop genres with NaN values in clustering features
cluster_data.dropna(inplace=True)

# Normalize the features for better clustering performance
scaler = StandardScaler()
cluster_data_scaled = scaler.fit_transform(cluster_data[['Price', 'Peak CCU']])

# Determine the optimal number of clusters using the Elbow Method
inertia = []
k_range = range(1, 11)  # Trying k from 1 to 10
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(cluster_data_scaled)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o')
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.show()

# Proceeding with k=4 (based on the elbow curve assumption)
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
cluster_data['Cluster'] = kmeans.fit_predict(cluster_data_scaled)

# Visualize the clusters
plt.figure(figsize=(10, 7))
for cluster in range(optimal_k):
    cluster_points = cluster_data[cluster_data['Cluster'] == cluster]
    plt.scatter(cluster_points['Price'], cluster_points['Peak CCU'], label=f'Cluster {cluster}')

plt.title("K-Means Clustering of Genres")
plt.xlabel("Average Price ($)")
plt.ylabel("Average Peak CCU")
plt.legend()
plt.grid(True)
plt.show()

# Display the cluster centroids and genres within each cluster
cluster_summary = cluster_data.groupby('Cluster')[['Price', 'Peak CCU']].mean()
print(cluster_summary)
print(cluster_data[['Genres', 'Cluster']])

#%%[markdown]

# #### Price and Popularity Insights:
#
#### Pricing Trends:
#
# Genres with niche appeal or low production costs (e.g., documentaries, free-to-play games) tend to be in the lower price clusters.
#
# Popular and immersive genres (e.g., RPG, Action, Strategy) justify higher prices due to higher development costs and a loyal audience.
#
#### Popularity (Peak CCU):
#
# Cluster 3 highlights that some tgenres, like "Photo Editing," can achieve extreme popularity despite moderate prices, indicating a unique value proposition.
#
# Cluster 2 confirms that mainstream genres (e.g., Adventure, Simulation) consistently attract large audiences.
#
# #### Business Strategy Recommendations:
#
#### Developers:
# Cluster 0: Focus on volume-driven monetization strategies, such as ad revenue or microtransactions, for low-cost, niche genres.
#
# Cluster 1: Maintain balance between affordability and feature richness to capture moderate audiences.
#
# Cluster 2: Invest in premium features, storytelling, or immersive gameplay to justify higher pricing and attract core gamers.
#
# Cluster 3: Leverage loyalty or niche appeal for tools or software genres to retain high engagement.
#

#### Consumers:
#
# Players on a budget can explore genres in Cluster 0 for affordable options.
#
# Hardcore gamers seeking premium experiences should target Cluster 2 genres.

#%% [markdown]
# ### 8.2 : Classification model - (K Nearest neighbors)
#
#### Comprehensive Analysis of K-Nearest Neighbors (KNN) Classification Model for price categorization and prediction

#%%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Encode genres using one-hot encoding
genres_encoded = pd.get_dummies(games_df_cleaned['Genres'], prefix='Genre')


X = pd.concat([genres_encoded, games_df_cleaned[['Peak CCU', 'Price']]], axis=1)
y = games_df_cleaned['Price Category']


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Standardize numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#%%

# Create and train the KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Predict the price category for the test set
y_pred = knn.predict(X_test_scaled)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Ensure y_test and y_pred are of type 'category'
y_test = y_test.astype('category')
y_pred = pd.Series(y_pred).astype('category')

#%%
from sklearn.metrics import accuracy_score

# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"Accuracy Score: {accuracy:.2f}%")



# %% [markdown]

# #### The KNN model's classification performance is outstanding, with near-perfect precision, recall, and F1-scores across all price categories (Low, Medium, High).
# Conclusion about the KNN Model:
# 
# Classification Report:
#
# The model achieves perfect precision, recall, and F1-scores (1.00) for all classes (Low, Medium, High).
# 
# The classes are balanced, and no single class dominates the dataset significantly.
#
# This indicates that the model is highly accurate in predicting all three price categories without any bias towards one class.
#
# The overall accuracy score is 99.9%, indicating perfect performance on the test data.
#
# Despite having high accuracy, the model is unlikely to over fit due to balanced classes and fewer misclassifications across different price groups
#
# Also, as KNN is a straightforward, non-parametric model that generally doesn't overfit unless the dataset is extremely small or noisy. This huge dataset should not be a concern.
#
# Hence, the low number of misclassifications suggests that the data may inherently have clear distinctions between classes, making KNN highly effective for this scenario.
#
# Model Evaluation:
#
# The KNN model has successfully utilized features like Genres, Price, and Peak CCU to classify price categories effectively.
#
# #### This model demonstrates that pricing and CCU, combined with genre data, are strong predictors of game success and cluster categorization.


# %% [markdown]
# ### Confusion Matrix Visualization for KNN Price Category Prediction

#%%

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=['Low', 'Medium', 'High'])

# Plot confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
plt.title('Confusion Matrix for KNN Model')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


#%%[markdown]

#### Diagonal Values (87178, 86885, 86620):
#
#### These are the True Positives (TP).
# The model correctly predicted:
#
# 87,178 instances as "Low" when the actual class was "Low."
#
# 86,885 instances as "Medium" when the actual class was "Medium."
# 86,620 instances as "High" when the actual class was "High."
#
# Off-Diagonal Values (2, 4, 16, 10, 3):
#
# These are Misclassifications (False Positives or False Negatives).


#%%[markdown] 
#### Conclusion for this research
#
#### 1. Pricing Trends in Genres and Categories:
# #### Genres with Higher Prices:
#### Genres like Simulation, Audio Production, and RPG command the highest average prices, indicating their specialized or immersive nature that justifies premium pricing.
#
#### These genres often cater to niche markets or involve significant development resources.
#
# #### Genres with Lower Prices:
#### Genres like Free to Play, Casual, and Documentary have the lowest average prices, highlighting their appeal to a broad audience and reliance on alternative revenue models (e.g., in-game purchases or ads).
#
# #### Categories with Higher Prices:
#### Categories like SteamVR Collectibles and Remote Play on TV reflect premium-priced features that are hardware or tech-intensive, showcasing the value placed on cutting-edge gaming experiences.
#
# #### Categories with Lower Prices:
#### Categories like In-App Purchases, Mods, and Cross-Platform Multiplayer often emphasize accessibility and inclusivity, leveraging lower price points to attract mass-market players.
#### Popularity vs. Pricing:
#
#### Genres like Massively Multiplayer and Action see high popularity (e.g., Peak CCU) despite moderate prices, indicating that these genres thrive on large-scale participation.
#
#### Free to Play games demonstrate a clear trend: lower upfront prices boost user engagement, leveraging volume for profitability.
#### Category-Genre Interplay:
#
#### The heatmap analysis showed clear variations in average pricing across different genre-category combinations.
#### Premium-priced genres like Audio Production and categories like SteamVR Collectibles align with tech-heavy or professional tools.


#%% [markdown]
# ### Recommendations
#
# #### For Developers:
#
# Premium Experiences: If you're making a niche or advanced game (e.g., VR or Simulation), you can set higher prices because players expect to pay more for unique or high-quality experiences.
#
# Mass-Market Games: For popular genres like Free to Play or Casual games, keep prices low or free and focus on earning money through in-game purchases or ads.
#
# For Players:
#
# Affordable Options: If you want budget-friendly games, look at categories like Free to Play or Casual games, which are designed for everyone and often cost less.
#
# High-End Games: If you're looking for immersive or advanced experiences, check out premium genres like Simulation or categories like VR-supported games.
#
#### This research analyzed the interplay of pricing, popularity, and game features across Steam’s game library, providing actionable insights for developers and players.

# %% [markdown]

### SMART Question 2: How has the release year affected the estimated number of owners for games on Steam, especially for those released within the last five years? Additionally, does the price of a game influence its number of owners differently depending on the release year?


# %% 
from datetime import datetime

# Convert 'Release date' and extract the year
games_df_cleaned['Release_Date'] = pd.to_datetime(games_df_cleaned['Release date'], errors='coerce')
games_df_cleaned['Release Year'] = games_df_cleaned['Release_Date'].dt.year

# Calculate average from 'Estimated owners'
games_df_cleaned['Estimated Owners'] = games_df_cleaned['Estimated owners'].apply(lambda x: np.mean([int(i.replace(',', '')) for i in x.split(' - ')]) if isinstance(x, str) else np.nan)
yearly_ownership = games_df_cleaned.groupby('Release Year')['Estimated Owners'].agg(['mean', 'count']).reset_index()

# Remove year 2025 from analysis
yearly_ownership = yearly_ownership[yearly_ownership['Release Year'] < 2025]

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.bar(yearly_ownership['Release Year'], yearly_ownership['mean'])
plt.title('Average Estimated Owners by Release Year')
plt.xlabel('Release Year')
plt.ylabel('Average Estimated Owners')

plt.subplot(1, 2, 2)
plt.plot(yearly_ownership['Release Year'], yearly_ownership['count'], marker='o')
plt.title('Number of Games Released by Year')
plt.xlabel('Release Year')
plt.ylabel('Number of Games')

plt.tight_layout()
plt.show()

# %% [markdown]
# %%
import matplotlib.ticker as ticker  

# Filter data for the last 5 years
current_year = datetime.now().year
last_5_years = yearly_ownership[yearly_ownership['Release Year'] >= (current_year - 4)]

# Filter data for the years 1997-2001
early_years = yearly_ownership[(yearly_ownership['Release Year'] >= 1997) & (yearly_ownership['Release Year'] <= 2001)]

# Print values for both sets
print("\nAnalysis of Games Released in Last 5 Years:")
print(last_5_years)

print("\nAnalysis of Games Released in 1997-2001:")
print(early_years)

plt.figure(figsize=(12, 6))

# Plot for the last 5 years
plt.bar(last_5_years['Release Year'], last_5_years['mean'], color='blue')
plt.title('Average Estimated Owners (Last 5 Years)')
plt.xlabel('Release Year')
plt.ylabel('Average Estimated Owners')
plt.xticks(last_5_years['Release Year'], rotation=45)

plt.tight_layout()
plt.show()

# Plot for the years 1997-2001
plt.figure(figsize=(8, 6))
plt.bar(early_years['Release Year'], early_years['mean'], color='green')
plt.title('Average Estimated Owners (1997-2001)')
plt.xlabel('Release Year')
plt.ylabel('Average Estimated Owners')
plt.xticks(early_years['Release Year'], rotation=45)

# Format y-axis to display actual values (Adjusting the scale)
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

plt.tight_layout()
plt.show()

# %% 

# Caverage game price
yearly_price_avg = games_df_cleaned.groupby('Release Year')['Price'].mean().reset_index()
yearly_combined = pd.merge(yearly_ownership, yearly_price_avg, on='Release Year')

plt.figure(figsize=(14, 7))

# Plot
plt.subplot(2, 1, 1)
plt.plot(yearly_combined['Release Year'], yearly_combined['mean'], color='dodgerblue', label='Avg. Owners')
plt.title('Average Estimated Owners by Release Year')
plt.xlabel('Release Year')
plt.ylabel('Avg. Estimated Owners')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(yearly_combined['Release Year'], yearly_combined['Price'], color='green', label='Avg. Price')
plt.title('Average Game Price by Release Year')
plt.xlabel('Release Year')
plt.ylabel('Avg. Price ($)')
plt.legend()

plt.tight_layout()
plt.show()

# Correlation 
correlation = yearly_combined[['mean', 'Price']].corr().iloc[0, 1]
print(f"Correlation between Average Price and Average Estimated Owners: {correlation:.2f}")


# %%

from scipy.stats import pearsonr

# Extract necessary columns and drop missing values
price_owners_df = games_df_cleaned[['Price', 'Estimated Owners']].dropna()

# Pearson correlation test
correlation_coefficient, p_value = pearsonr(price_owners_df['Price'], price_owners_df['Estimated Owners'])

print(f"Pearson Correlation Coefficient: {correlation_coefficient:.2f}")
print(f"P-value: {p_value:.4f}")
if p_value < 0.05:
    print("There is a statistically significant association between game price and the number of owners.")
else:
    print("There is no statistically significant association between game price and the number of owners.")

# plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=price_owners_df, x='Price', y='Estimated Owners', alpha=0.6)
plt.title('Scatter Plot of Game Price vs. Estimated Owners')
plt.xlabel('Price ($)')
plt.ylabel('Estimated Owners')
plt.xscale('log')  
plt.yscale('log')
plt.grid(True)
plt.show()

# %% [markdown] 
## SMART QUESTION 3
## Which games and game categories consistently reach the highest peak concurrent users, and does this trend differ significantly across genres?


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
plt.yscale('log') 
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

g.set(yscale='log')
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
plt.yscale('log')
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
categorical_features = ['Categories', 'Genres', 'Developers', 'Publishers']

# Numerical features
numerical_features = ['Price']

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

from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(n_estimators=200, max_depth=10, learning_rate=0.05, random_state=42))
])

xgb_pipeline.fit(X_train, y_train)

y_pred = xgb_pipeline.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"XGBoost Model: MSE = {mse:.2f}, R2 = {r2:.2f}")

# %% [markdown]

#### Hyperparameter tuning 

from sklearn.model_selection import GridSearchCV

# Parameter grid for XGBoost
param_grid = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__max_depth': [6, 10, 15, 20],
    'regressor__learning_rate': [0.01, 0.05, 0.1]
}

# Grid search
grid_search = GridSearchCV(xgb_pipeline, param_grid, cv=3, scoring='r2', verbose=2)
grid_search.fit(X_train, y_train)

# Best parameters
print(f"Best Parameters: {grid_search.best_params_}")

# Evaluating the best model
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
mse_best = mean_squared_error(y_test, y_pred_best)
r2_best = r2_score(y_test, y_pred_best)

print(f"Best Tuned Model: MSE = {mse_best:.2f}, R2 = {r2_best:.2f}")


# %% [markdown]

#### LightGBM Model

from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

lgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LGBMRegressor(n_estimators=200, max_depth=10, learning_rate=0.05, random_state=42))
])

# Train the model
lgb_pipeline.fit(X_train, y_train)

y_pred_lgb = lgb_pipeline.predict(X_test)

# Evaluating the model
mse_lgb = mean_squared_error(y_test, y_pred_lgb)
r2_lgb = r2_score(y_test, y_pred_lgb)

print(f"LightGBM Model: MSE = {mse_lgb:.2f}, R2 = {r2_lgb:.2f}")

# %% [markdown]


#### Hyperparameter Tuning

from sklearn.model_selection import GridSearchCV

# Parameter grid for LightGBM
param_grid_lgb = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__max_depth': [6, 10, 15],
    'regressor__learning_rate': [0.01, 0.05, 0.1],
    'regressor__num_leaves': [31, 50, 100]
}

# Grid search
grid_search_lgb = GridSearchCV(lgb_pipeline, param_grid_lgb, cv=3, scoring='r2', verbose=2)
grid_search_lgb.fit(X_train, y_train)

# Best parameters
print(f"Best Parameters: {grid_search_lgb.best_params_}")

best_lgb_model = grid_search_lgb.best_estimator_
y_pred_best_lgb = best_lgb_model.predict(X_test)
mse_best_lgb = mean_squared_error(y_test, y_pred_best_lgb)
r2_best_lgb = r2_score(y_test, y_pred_best_lgb)

print(f"Best Tuned LightGBM Model: MSE = {mse_best_lgb:.2f}, R2 = {r2_best_lgb:.2f}")





# %% [markdown]
### 7.0 SMART Question 4 - How have game releases evolved over time across genres, and which genres have shown the highest growth particularly focusing on the top 10 genres from 2014 to 2024?

games_df_cleaned = games_df_cleaned.dropna(subset=['Release date'])
games_df_cleaned['Release date'] = pd.to_datetime(games_df_cleaned['Release date'], errors='coerce')

games_df_cleaned['Year_released'] = games_df_cleaned['Release date']
games_df_cleaned['Year'] = games_df_cleaned['Year_released'].dt.year
games_df_cleaned['Month'] = games_df_cleaned['Year_released'].dt.month
games_df_cleaned['Day'] = games_df_cleaned['Year_released'].dt.day

# Cyclical encoding for month and day
games_df_cleaned['Month_sin'] = np.sin(2 * np.pi * games_df_cleaned['Month'] / 12)
games_df_cleaned['Month_cos'] = np.cos(2 * np.pi * games_df_cleaned['Month'] / 12)
games_df_cleaned['Day_sin'] = np.sin(2 * np.pi * games_df_cleaned['Day'] / 31)
games_df_cleaned['Day_cos'] = np.cos(2 * np.pi * games_df_cleaned['Day'] / 31)


print(games_df_cleaned.head())

games_df_cleaned =  games_df_cleaned[(games_df_cleaned['Year'] >= 2014) & (games_df_cleaned['Year'] <= 2024)]

games_df_cleaned.head()


# %% [markdown]
# # Splitting and Formatting Genres
games_df_cleaned['Genres'] = games_df_cleaned['Genres'].str.split(',')
games_df_cleaned['Genres'] = games_df_cleaned['Genres'].apply(lambda x: [a.lower().title() for a in x])

unique_genres = set()
for genres in games_df_cleaned['Genres']:
    unique_genres.update(genres)
    
len(unique_genres)



# %%
unique_genres
# %% [markdown]
# # Filtering Unwanted Genres
genres_to_remove = ["Free To Play","Early Access"]

unique_genres = unique_genres.difference(genres_to_remove)
len(unique_genres)

# %% 
games_df_cleaned.head()



# %%
columns = ["Year_released"]
columns.extend(list(unique_genres))

genres_by_year = pd.DataFrame(columns=columns)
genres_by_year 



# %%[markdown]
# # Creating a Dictionary of Genres by Year
genres_year_dict = games_df_cleaned.groupby(games_df_cleaned["Year_released"]).apply(dict,include_groups=False).to_dict()
genres_year_dict



# %% [markdown]
# # Game Releases by Year and Genre
exploded_df = games_df_cleaned.explode('Genres')
release_trends = exploded_df.groupby(['Year_released', 'Genres']).size().unstack(fill_value=0)

print("Game Releases by Year and Genre:")
print(release_trends)




# %% [markdown]
# ### Identifying Top 10 Genres
genre_counts = exploded_df['Genres'].value_counts()

top_10_genres = genre_counts.head(10).index.tolist()
print("Top 10 Genres:")
print(top_10_genres)


# %% [markdown]
# ###Explode Tags column for individual genre-level analysis
top_10_genres_df = exploded_df[exploded_df['Genres'].isin(top_10_genres)]
release_trends_top10 = top_10_genres_df.groupby(['Year', 'Genres']).size().unstack(fill_value=0)

print("Year-wise Trends for Top 10 Genres:")
print(release_trends_top10)


# %% [markdown]
# ### Visualizing Game Releases by Genre Over Time
exploded_df = top_10_genres_df.explode('Genres')

# Group by Year and Genre to analyze release trends
release_trends = exploded_df.groupby(['Year', 'Genres']).size().unstack(fill_value=0)

yearly_totals = release_trends.sum(axis=1)

# Plot the stacked bar chart
ax = release_trends.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='tab20')

plt.title('Game Releases Over Time by Genre')
plt.xlabel('Year Released')
plt.ylabel('Number of Games Released')
plt.legend(title='Genres', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Add percentage labels on each bar
for i, year in enumerate(release_trends.index):
    total = yearly_totals[year]
    for genre in release_trends.columns:
        count = release_trends.loc[year, genre]
        if count > 0:
            percentage = (count / total) * 100
            ax.text(i, count / 2 + release_trends.loc[year].cumsum()[genre] - count,
                    f'{percentage:.1f}%', ha='center', va='center', color='white', fontsize=8)

plt.show()

# %% [markdown]
# ### Proportional Trends of Top 10 Genres Over Time
import plotly.graph_objects as go

release_trends_normalized = release_trends_top10.div(release_trends_top10.sum(axis=1), axis=0)

fig = go.Figure()
for genre in release_trends_normalized.columns:
    fig.add_trace(go.Scatter(
        x=release_trends_normalized.index,
        y=release_trends_normalized[genre],
        mode='lines',
        stackgroup='one',
        name=genre
    ))

fig.update_layout(
    title='Proportion of Games Released by Top 10 Genres Over Time',
    xaxis_title='Year Released',
    yaxis_title='Proportion of Games',
    showlegend=True
)
fig.show()



# %%[markdown]
# ### Calculating Growth Rates by Genre

initial_counts = release_trends_top10.iloc[0] 
final_counts = release_trends_top10.iloc[-1]

growth_rates = ((final_counts - initial_counts) / initial_counts.replace(0, np.nan)) * 100
growth_rates = growth_rates.fillna(0).sort_values(ascending=False)

print("Growth Rates by Genre:")
print(growth_rates)

# %% [markdown]
# ### Determining Top Genre Each Year
release_trends_top10['Top Genre'] = release_trends_top10.idxmax(axis=1)

top_genre_counts = release_trends_top10['Top Genre'].value_counts()

print("Top Genre Each Year:")
print(release_trends_top10[['Top Genre']])
print("\nTop Genre Counts:")
print(top_genre_counts)



# %%[markdown]
# # Model Evaluation for Genre Prediction Using Random Forest
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


genre_accuracy_results = {}
one_hot_tags = games_df_cleaned['Genres'].apply(lambda x: pd.Series(1, index=x)).fillna(0)
features = pd.concat([games_df_cleaned[['Year', 'Month', 'Day']], one_hot_tags], axis=1)
for genre in top_10_genres:
    print(f"\nEvaluating model for genre: {genre}")
    

    features['Target'] = one_hot_tags[genre]
    

    X = features[['Year', 'Month', 'Day']] 
    y = features['Target']                 
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Accuracy for {genre}: {accuracy * 100:.2f}%")
    print(f"Classification Report for {genre}:\n{report}")
    
    genre_accuracy_results[genre] = {
        'accuracy': accuracy,
        'classification_report': report
    }

print("\nSummary of Accuracy Results for Top 10 Genres:")
for genre, results in genre_accuracy_results.items():
    print(f"{genre}: Accuracy = {results['accuracy'] * 100:.2f}%")



# %% [markdown]
# ### Forecasting Future Trends with Prophet
from prophet import Prophet

if isinstance(games_df_cleaned['Genres'].iloc[0], list):
    games_df_cleaned = games_df_cleaned.explode('Genres')

games_df_cleaned['Year'] = games_df_cleaned['Release date'].dt.year
genre_monthly_trends = games_df_cleaned.groupby(['Year', 'Month', 'Genres']).size().unstack(fill_value=0)
specified_genres = ["Early Access", "Free To Play", "Casual", "Simulation", 
                    "Sports", "Adventure", "Indie", "Rpg", "Strategy", "Action"]
genre_monthly_trends = genre_monthly_trends[specified_genres]
future_monthly_trends = {}
months_future = pd.date_range(start='2025-01-01', periods=12, freq='M')

for genre in genre_monthly_trends.columns:
    last_year_data = genre_monthly_trends[genre].iloc[-12:].values
    increment = np.mean(last_year_data) * 0.05  
    future_data = last_year_data + increment
    future_monthly_trends[genre] = future_data[:12]  

# %% [markdown]
# # Preparing Data for Forecasting with Prophet
if isinstance(games_df_cleaned['Genres'].iloc[0], list):
    games_df_cleaned = games_df_cleaned.explode('Genres')

# Add Year and Month columns
games_df_cleaned['Year'] = games_df_cleaned['Release date'].dt.year
games_df_cleaned['Month'] = games_df_cleaned['Release date'].dt.month

# Aggregate data by year, month, and genre
genre_monthly_trends = games_df_cleaned.groupby(['Year', 'Month', 'Genres']).size().unstack(fill_value=0)


specified_genres = ["Casual", "Simulation", 
                    "Sports", "Adventure", "Indie", "Rpg", "Strategy", "Action"]
genre_monthly_trends = genre_monthly_trends[specified_genres]

forecasts = {}
for genre in specified_genres:
    df_genre = genre_monthly_trends[[genre]].reset_index()
    df_genre['ds'] = pd.to_datetime(df_genre['Year'].astype(str) + '-' + df_genre['Month'].astype(str))
  

# %% [markdown]
# # Forecasting Monthly Game Releases Using Prophet
filtered_data = exploded_df[exploded_df['Genres'].isin(specified_genres)]
monthly_data = exploded_df.groupby(['Year', 'Month', 'Genres']).size().unstack(fill_value=0).reset_index()

from prophet import Prophet

forecasts = {}

forecasts = {}

for genre in specified_genres:

    genre_data = monthly_data[[genre]].reset_index()
    genre_data['ds'] = df_genre['ds']
    genre_data = genre_data.rename(columns={'Date': 'ds', genre: 'y'}) 

    print(f"Prepared data for genre: {genre}")
    print(genre_data.head()) 
    

    model = Prophet()
    model.fit(genre_data)
    

    future = model.make_future_dataframe(periods=24, freq='M') 
    forecast = model.predict(future)
    
    forecasts[genre] = forecast[['ds', 'yhat']].set_index('ds')

combined_forecasts = pd.DataFrame()

for genre, forecast in forecasts.items():
    forecast = forecast.rename(columns={'yhat': genre})
    if combined_forecasts.empty:
        combined_forecasts = forecast
    else:
        combined_forecasts = combined_forecasts.join(forecast, how='outer')


combined_forecasts = combined_forecasts['2014-01-01':'2025-12-31']
print(combined_forecasts.head()) 
# %% [markdown]
# # Visualizing Predicted Game Releases by Genre (2025-2026)
import plotly.graph_objects as go

fig = go.Figure()

for genre in combined_forecasts.columns:
    fig.add_trace(go.Scatter(
        x=combined_forecasts.index,
        y=combined_forecasts[genre],
        mode='lines',
        stackgroup='one',  # Stacked area chart
        name=genre
    ))

fig.update_layout(
    title='Predicted Game Releases by Genre (2014-2025)',
    xaxis_title='Date',
    yaxis_title='Number of Predicted Releases',
    showlegend=True
)
fig.show()
# %%
import pandas as pd
import numpy as np
from scipy.stats import f_oneway

# Ensure 'Release date' is in datetime format
games_df_cleaned['Release date'] = pd.to_datetime(games_df_cleaned['Release date'], errors='coerce')

# Extract year from 'Release date' to use as a numeric value for ANOVA
games_df_cleaned['Year'] = games_df_cleaned['Release date'].dt.year

# Explode the genres to have one genre per row
exploded_df = games_df_cleaned.explode('Genres')

# Filter for the top 10 genres if needed (optional)
top_10_genres = exploded_df['Genres'].value_counts().head(10).index.tolist()
exploded_df = exploded_df[exploded_df['Genres'].isin(top_10_genres)]

# Group data by genre and collect years of release
genre_release_years = exploded_df.groupby('Genres')['Year'].apply(list)

# Perform ANOVA test
anova_results = f_oneway(*genre_release_years)

# Output the results
print(f"ANOVA F-statistic: {anova_results.statistic}")
print(f"ANOVA p-value: {anova_results.pvalue}")

# Interpret the results
if anova_results.pvalue < 0.05:
    print("There is a significant difference in release years across genres.")
else:
    print("There is no significant difference in release years across genres.")


# %%
