import pandas as pd
from flask import Flask, render_template, request
import pickle
from textblob import TextBlob
import matplotlib.pyplot as plt
import io
import base64
from dashboard import create_dashboard

app = Flask(__name__)

try:
    reviews_data = pd.read_csv('dataset_pfa.csv', encoding='utf-8')
    reviews_data['reviews.rating'] = reviews_data['reviews.rating'] / 2.0  # Normalize ratings to 0-5
except FileNotFoundError:
    print("Error: Dataset file 'dataset_pfa.csv' not found.")
    reviews_data = pd.DataFrame()

try:
    reviews_data_count = pd.read_csv('reviews_data_count.csv', encoding='utf-8')
except FileNotFoundError:
    print("Error: Dataset file 'dataset_pfa.csv' not found.")
    reviews_data = pd.DataFrame()


def sentiment_counts_to_csv(dataframe, output_path):
    """
    Calcule le nombre de ratings positifs, neutres et négatifs pour chaque hôtel,
    puis enregistre les résultats dans un fichier CSV.

    Args:
        dataframe (pd.DataFrame): Un DataFrame contenant les colonnes 'name' (nom des hôtels)
                                  et 'reviews.rating' (notes des avis).
        output_path (str): Le chemin complet où enregistrer le fichier CSV.

    Returns:
        None
    """
    # Vérifier si les colonnes nécessaires existent
    if 'name' not in dataframe.columns or 'reviews.rating' not in dataframe.columns:
        raise ValueError("Le DataFrame doit contenir les colonnes 'name' et 'reviews.rating'")

    # Catégoriser les ratings
    dataframe['Sentiment'] = dataframe['reviews.rating'].apply(
        lambda x: 'Positif' if x > 3 else ('Négatif' if x < 3 else 'Neutre')
    )

    # Calculer le compte des sentiments pour chaque hôtel
    sentiment_counts = dataframe.groupby('name')['Sentiment'].value_counts().unstack(fill_value=0)

    # Enregistrer les résultats dans un fichier CSV
    sentiment_counts.to_csv(output_path)
    print(f"Les résultats ont été enregistrés dans le fichier : {output_path}")


def load_model_and_vectorizer(model_filename='sentiment_analysis_model.pkl', vectorizer_filename='vectorizer.pkl'):
    try:
        model = pickle.load(open(model_filename, 'rb'))
        vectorizer = pickle.load(open(vectorizer_filename, 'rb'))
        return model, vectorizer
    except FileNotFoundError:
        print(f"Error: One or both of the files ({model_filename}, {vectorizer_filename}) not found.")
        return None


def clean_review(review):
    return ' '.join(TextBlob(review).words)


def predict_sentiment(review, model, vectorizer):
    if model is None or vectorizer is None:
        print("Error: Model or vectorizer not loaded.")
        return None
    cleaned_review = clean_review(review)
    review_vec = vectorizer.transform([cleaned_review])
    prediction = model.predict(review_vec)
    res = str(prediction[0])
    return res


# Charger le modèle et le vectorizer
model, vectorizer = load_model_and_vectorizer()


@app.route('/')
def hello_usr():
    return render_template('indexx.html')


@app.route('/submit', methods=['GET', 'POST'])
def handle_user_choice():
    # Charger les données des hôtels
    hotels = pd.read_csv('unique_hotels_with_avg_ratings_and_comments.csv', encoding='utf-8')
    hotels.set_index('name', inplace=True)

    sentiment_result = None  # Initialiser la variable pour le résultat d'analyse
    pie_chart_url = None     # Initialiser la variable pour le camembert

    if request.method == 'POST':
        user_choice = request.form.get('hotel')
        print("User choice:", user_choice)

        if user_choice in hotels.index:
            # Obtenir un exemple de commentaire pour l'analyse
            com = hotels.loc[user_choice]
            comment = com['sample_comment']
            sentiment_result = predict_sentiment(comment, model, vectorizer)

            # Générer le camembert à partir des données de sentiment
            try:
                # Charger les données des sentiments
                sentiment_data = pd.read_csv('reviews_data_count.csv', encoding='utf-8')
                sentiment_data.set_index('name', inplace=True)

                if user_choice in sentiment_data.index:
                    sentiments = sentiment_data.loc[user_choice]
                    labels = sentiments.index
                    values = sentiments.values

                    # Créer le camembert
                    plt.figure(figsize=(6, 6))
                    plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#76c7c0', '#f5c518', '#e74c3c'])
                    plt.title(f"Répartition des sentiments pour {user_choice}")

                    # Sauvegarder dans un flux mémoire
                    img = io.BytesIO()
                    plt.savefig(img, format='png', bbox_inches='tight')
                    img.seek(0)
                    pie_chart_url = base64.b64encode(img.getvalue()).decode()
                    plt.close()

            except Exception as e:
                print("Erreur lors de la génération du camembert :", e)
                sentiment_result = "Erreur lors de la génération des données."

        else:
            print("Hotel not found:", user_choice)
            sentiment_result = "Hôtel non trouvé."

    return render_template(
        'indexx.html',
        sentiment=sentiment_result,
        pie_chart_url=pie_chart_url,
        scroll_to_result=True
    )



@app.route('/submit_comment', methods=['POST'])
def handle_comment():
    global reviews_data  # Utiliser le dataframe global

    # Récupérer les données envoyées par le formulaire
    hotel_name = request.form.get('hotel')  # Nom de l'hôtel
    rating = request.form.get('rating')  # Évaluation (1 à 5)
    comment = request.form.get('userComment')  # Commentaire

    if not hotel_name or not rating or not comment:
        return render_template(
            'indexx.html',
            sentiment="Veuillez remplir tous les champs pour soumettre votre avis.",
            scroll_to_result=True
        )

    # Ajouter l'entrée dans le dataframe
    new_entry = {
        "name": hotel_name,
        "reviews.rating": float(rating),  # Note convertie en float
        "reviews.text": comment,  # Texte du commentaire
        "reviews.dateAdded": pd.Timestamp.now()  # Date d'ajout
    }

    reviews_data = pd.concat([reviews_data, pd.DataFrame([new_entry])], ignore_index=True)

    # Sauvegarder les données mises à jour dans un fichier CSV
    reviews_data.to_csv('updated_reviews_data.csv', index=False, encoding='utf-8')
    sentiment_counts_to_csv(reviews_data, 'reviews_data_count.csv',)

    return render_template(
        'indexx.html',
        sentiment="Merci pour votre avis! Vos données ont été enregistrées.",
        scroll_to_result=True
    )


var = create_dashboard(app)

if __name__ == '__main__':
    app.run(debug=True)
