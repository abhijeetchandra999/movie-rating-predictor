from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the scaler and model
scaler = joblib.load("scaler.pkl")
model = joblib.load("knn_model.pkl")

# Predefined genres (must match checkboxes in HTML)
genre_list = [
    "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary", "Drama",
    "Family", "Fantasy", "Foreign", "History", "Horror", "Music", "Mystery",
    "Romance", "Science Fiction", "TV Movie", "Thriller", "War", "Western"
]

@app.route('/')
def home():
    return render_template('index.html', genres=genre_list)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = []

        # Numeric inputs
        budget = float(request.form['budget'])
        popularity = float(request.form['popularity'])
        vote_average = float(request.form['vote_average'])
        vote_count = float(request.form['vote_count'])
        features.extend([budget, popularity, vote_average, vote_count])

        # Binary categorical inputs
        yes_no_map = {'Yes': 1, 'No': 0}
        features.append(yes_no_map[request.form['is_english']])
        features.append(yes_no_map[request.form['is_released']])
        features.append(yes_no_map[request.form['has_company']])
        features.append(yes_no_map[request.form['has_country']])
        features.append(yes_no_map[request.form['is_director_top10']])
        features.append(yes_no_map[request.form['has_actor_top10']])

        # Handle genres
        selected_genres = request.form.getlist('genres')  # List of selected genres
        genre_features = [1 if genre in selected_genres else 0 for genre in genre_list]
        features.extend(genre_features)

        # Convert and scale
        final_features = np.array(features).reshape(1, -1)
        final_scaled = scaler.transform(final_features)
        prediction = model.predict(final_scaled)

        return render_template('index.html', genres=genre_list, prediction_text=f"Predicted Rating Category: {prediction[0]}")

    except Exception as e:
        return render_template('index.html', genres=genre_list, prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
