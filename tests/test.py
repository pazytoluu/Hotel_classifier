# prompt: Je veux un dataframe avec uniquement les noms des hotels dans l'ordre alphabetique sans les doublons  avec la moyenne des rating et un sample de commentaire comme ça name,city,avg_rating,sample_comment

import pandas as pd

# Assuming 'dataset_pfa.csv' is in the current directory or provide the correct path
data = pd.read_csv('../dataset_pfa.csv', encoding='UTF-8')

# Group by hotel name and calculate the average rating
avg_ratings = data.groupby('name')['reviews.rating'].mean().reset_index()
avg_ratings.rename(columns={'reviews.rating': 'avg_rating'}, inplace=True)

# Get a sample comment for each hotel
sample_comments = data.groupby('name')['reviews.text'].first().reset_index()
sample_comments.rename(columns={'reviews.text': 'sample_comment'}, inplace=True)

# Merge the dataframes
hotel_data = pd.merge(avg_ratings, sample_comments, on='name')


# Get city for each hotel
hotel_cities = data.groupby('name')['city'].first().reset_index()
hotel_data = pd.merge(hotel_data, hotel_cities, on='name')

# Sort hotels alphabetically by name and remove duplicates
hotel_data.sort_values('name', inplace=True)
hotel_data.drop_duplicates(subset=['name'], inplace=True)

#Select columns
hotel_data = hotel_data[['name', 'city', 'avg_rating', 'sample_comment']]

hotel_data.to_csv('unique_hotels_with_avg_ratings_and_comments.csv', index=False)