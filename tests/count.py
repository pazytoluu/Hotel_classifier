import pandas as pd

# Charger le fichier CSV
file_path = "updated_reviews_data.csv"
data = pd.read_csv(file_path)

# Vérifier si les colonnes "name" et "reviews.rating" existent
if "name" in data.columns and "reviews.rating" in data.columns:
    # Catégoriser les ratings
    data['Sentiment'] = data['reviews.rating'].apply(
        lambda x: 'Positif' if x > 3.5 else ('Négatif' if x < 2.5 else 'Neutre')
    )

    # Calculer le compte des sentiments pour chaque hôtel
    sentiment_counts_per_hotel = data.groupby('name')['Sentiment'].value_counts().unstack(fill_value=0)

    # Afficher les résultats
    print(sentiment_counts_per_hotel)
    sentiment_counts_per_hotel.to_csv('reviews_data_count.csv')
else:
    print("Les colonnes 'name' ou 'reviews.rating' sont introuvables dans le fichier CSV.")
