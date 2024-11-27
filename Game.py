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

# %% [markdown]
# ## 6. Final Check for Missing Values After Cleaning

# After cleaning, check if there are any remaining missing values in the cleaned dataset.
print(games_df_cleaned.isnull().sum())
games_df_cleaned.shape


# %% [markdown]
# ## 7.0 SMART Question 4 - How have game releases evolved over time across genres, 
# and which genres have shown the highest growth, consistent dominance, 
# and significant changes in their proportions and total releases, 
# particularly focusing on the top 10 genres from 2014 to 2024?

games_df_cleaned = games_df_cleaned.dropna(subset=['Release date'])
games_df_cleaned['Release date'] = pd.to_datetime(games_df_cleaned['Release date'], errors='coerce')

# Extract the year and create a new column 'Year_released'
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


# %%
games_df_cleaned['Tags'] = games_df_cleaned['Tags'].str.split(',')
games_df_cleaned['Tags'] = games_df_cleaned['Tags'].apply(lambda x: [a.lower().title() for a in x])

unique_genres = set()
for genres in games_df_cleaned['Tags']:
    unique_genres.update(genres)
    
len(unique_genres)



# %%
unique_genres
# %%
genres_to_remove = ["Free To Play","Early Access"]
unique_genres = unique_genres.difference(genres_to_remove)
len(unique_genres)




# %%
columns = ["Year_released"]
columns.extend(list(unique_genres))

genres_by_year = pd.DataFrame(columns=columns)
genres_by_year 



# %%
genres_year_dict = games_df_cleaned.groupby(games_df_cleaned["Year_released"]).apply(dict,include_groups=False).to_dict()
genres_year_dict



# %%
exploded_df = games_df_cleaned.explode('Tags')
release_trends = exploded_df.groupby(['Year_released', 'Tags']).size().unstack(fill_value=0)

print("Game Releases by Year and Genre:")
print(release_trends)



# %%
import matplotlib.pyplot as plt

release_trends.plot(kind='bar', stacked=True, figsize=(12, 6))
plt.title('Game Releases Over Time by Genre')
plt.xlabel('Year Released')
plt.ylabel('Number of Games Released')
plt.legend(title='Genres', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# %%
genre_counts = exploded_df['Tags'].value_counts()

top_10_genres = genre_counts.head(10).index.tolist()
print("Top 10 Genres:")
print(top_10_genres)


# %%
top_10_genres_df = exploded_df[exploded_df['Tags'].isin(top_10_genres)]
release_trends_top10 = top_10_genres_df.groupby(['Year', 'Tags']).size().unstack(fill_value=0)

print("Year-wise Trends for Top 10 Genres:")
print(release_trends_top10)


# %%
# Explode Tags column for individual genre-level analysis
exploded_df = top_10_genres_df.explode('Tags')

# Group by Year and Tags to analyze release trends
release_trends = exploded_df.groupby(['Year', 'Tags']).size().unstack(fill_value=0)

# Plot stacked bar chart for game releases by genre over time
release_trends.plot(kind='bar', stacked=True, figsize=(12, 6))
plt.title('Game Releases Over Time by Genre')
plt.xlabel('Year Released')
plt.ylabel('Number of Games Released')
plt.legend(title='Genres', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# %%
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



# %%

initial_counts = release_trends_top10.iloc[0] 
final_counts = release_trends_top10.iloc[-1]

growth_rates = ((final_counts - initial_counts) / initial_counts.replace(0, np.nan)) * 100
growth_rates = growth_rates.fillna(0).sort_values(ascending=False)

print("Growth Rates by Genre:")
print(growth_rates)

# %%
release_trends_top10['Top Genre'] = release_trends_top10.idxmax(axis=1)

top_genre_counts = release_trends_top10['Top Genre'].value_counts()

print("Top Genre Each Year:")
print(release_trends_top10[['Top Genre']])
print("\nTop Genre Counts:")
print(top_genre_counts)



# %%
from sklearn.metrics import accuracy_score, classification_report

# Initialize a dictionary to store results for each genre
genre_accuracy_results = {}
one_hot_tags = games_df_cleaned['Tags'].apply(lambda x: pd.Series(1, index=x)).fillna(0)
features = pd.concat([games_df_cleaned[['Year', 'Month', 'Day']], one_hot_tags], axis=1)
# Iterate through each genre in the top 10 list
for genre in top_10_genres:
    print(f"\nEvaluating model for genre: {genre}")
    
    # Create a target column for the current genre
    features['Target'] = one_hot_tags[genre]
    
    # Define features (X) and target (y)
    X = features[['Year', 'Month', 'Day']]  # Use Year, Month, Day as predictors
    y = features['Target']                  # Target is whether the game belongs to this genre
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate accuracy and classification report
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    # Print results for the current genre
    print(f"Accuracy for {genre}: {accuracy * 100:.2f}%")
    print(f"Classification Report for {genre}:\n{report}")
    
    # Store results in dictionary
    genre_accuracy_results[genre] = {
        'accuracy': accuracy,
        'classification_report': report
    }

# Print summary of results for all genres
print("\nSummary of Accuracy Results for Top 10 Genres:")
for genre, results in genre_accuracy_results.items():
    print(f"{genre}: Accuracy = {results['accuracy'] * 100:.2f}%")


