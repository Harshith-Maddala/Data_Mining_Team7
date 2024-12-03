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
# #### Understanding of genre-based pricing trends through clustering and classification model.

#%% [markdown]
# ### 8.1 : Clustering: Grouping Genres Based on Similarity ( K-Means Clustering )

#%% [markdown]
# the genres have been grouped into 4 clusters based on their average price and average Peak CCU (popularity):
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
# the classes are balanced, and no single class dominates the dataset significantly.
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

# %%
