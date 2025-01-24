import dash
from dash import dcc, html
import pandas as pd
import plotly.express as px

# Charger les données des hôtels
df = pd.read_csv("dataset_pfa.csv", encoding="UTF-8")

# Pré-traitement des données
# Remplacer les valeurs manquantes dans `reviews.rating` par 0 ou une autre stratégie (ex : moyenne).
df['reviews.rating'] = pd.to_numeric(df['reviews.rating'], errors='coerce')
df.dropna(subset=['reviews.rating'], inplace=True)

# Calculer la note moyenne par hôtel
average_ratings = df.groupby('name')['reviews.rating'].mean().reset_index()
average_ratings.rename(columns={'reviews.rating': 'average_rating'}, inplace=True)
average_ratings['average_rating'] = average_ratings['average_rating'] / 2


# Initialiser l'application Dash
def create_dashboard(flask_app):
    dash_app = dash.Dash(
        __name__,
        server=flask_app,
        url_base_pathname='/dashboard/',  # Lien pour accéder au dashboard
    )

    # Mise en page du dashboard
    dash_app.layout = html.Div([
        html.H1("Dashboard des Hôtels"),

        # Nombre total d'hôtels
        html.Div([
            html.H3("Nombre total d'hôtels"),
            html.P(f"{df['name'].nunique()}"),
        ], style={"margin-bottom": "20px"}),

        dcc.Graph(
            id="average-ratings",
            figure=px.bar(
                average_ratings.sort_values(by="average_rating", ascending=False).head(10),
                x="name",
                y="average_rating",
                title="Top 10 des hôtels par note moyenne",
                labels={"name": "Nom de l'hôtel", "average_rating": "Note moyenne"},
                text="average_rating",
                color="average_rating",  # Ajoute des couleurs dynamiques en fonction des valeurs
                color_continuous_scale="Viridis"  # Choisit un dégradé de couleurs
            )
        ),


        # Répartition des notes
        dcc.Graph(
            id="ratings-distribution",
            figure=px.histogram(
                df,
                x="reviews.rating",
                title="Répartition des notes moyennes",
                labels={"reviews.rating": "Note moyenne", "count": "Nombre d'hôtels"},
                nbins=10
            )
        ),

        # Répartition géographique des hôtels (par ville)
        dcc.Graph(
            id="hotel-locations",
            figure=px.scatter_mapbox(
                df,
                lat="latitude",
                lon="longitude",
                hover_name="name",
                hover_data={"city": True, "categories": True},
                color_discrete_sequence=["blue"],
                zoom=3,
                height=500,
                title="Localisation des Hôtels"
            ).update_layout(mapbox_style="open-street-map")
        ),
    ])

    return dash_app
