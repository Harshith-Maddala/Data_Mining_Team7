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

### SMART Question 2: How has the release year impacted the estimated number of owners for games on Steam, and are games released within the last five years have more estimated number of owners on average?

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

# %% One-way Anova Test
from scipy.stats import f_oneway

anova_data = games_df_cleaned[['Release Year', 'Estimated Owners']].dropna()
owners_by_year = [group['Estimated Owners'].values for year, group in anova_data.groupby('Release Year')]

f_stat, p_value = f_oneway(*owners_by_year)

print("One-Way ANOVA Test Results:")
print(f"F-statistic: {f_stat:.2f}")
print(f"P-value: {p_value:.12f}")
