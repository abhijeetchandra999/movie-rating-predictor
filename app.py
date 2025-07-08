from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# Load the scaler and model
with open("scaler.pkl", "rb") as f:
    scaler = joblib.load(f)

with open("knn_model.pkl", "rb") as f:
    model = joblib.load(f)

# Load top 10 metadata for display (optional, static display)
top_actors = ["Actor 1", "Actor 2", "Actor 3", "Actor 4", "Actor 5", "Actor 6", "Actor 7", "Actor 8", "Actor 9", "Actor 10"]
top_directors = ["Director 1", "Director 2", "Director 3", "Director 4", "Director 5", "Director 6", "Director 7", "Director 8", "Director 9", "Director 10"]
top_companies = ["Company 1", "Company 2", "Company 3", "Company 4", "Company 5", "Company 6", "Company 7", "Company 8", "Company 9", "Company 10"]
top_countries = ["USA", "UK", "India", "Canada", "France", "Germany", "Japan", "Australia", "China", "Spain"]

# Home route
@app.route("/")
def index():
    return render_template("index.html", 
                           actors=top_actors, 
                           directors=top_directors, 
                           companies=top_companies, 
                           countries=top_countries)

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Fetch form values
        features = []

        # Example: numeric inputs
        budget = float(request.form["budget"])
        popularity = float(request.form["popularity"])
        vote_average = float(request.form["vote_average"])
        vote_count = float(request.form["vote_count"])
        runtime = float(request.form["runtime"])

        features.extend([budget, popularity, vote_average, vote_count, runtime])

        # Example: binary inputs
        is_english = 1 if request.form.get("is_english") == "yes" else 0
        is_released = 1 if request.form.get("is_released") == "yes" else 0

        features.extend([is_english, is_released])

        # Add your One-Hot/Binary encoded values here for actors, companies, etc.
        # Example: top actor
        for actor in top_actors:
            features.append(1 if request.form.get("actor_name") == actor else 0)

        for director in top_directors:
            features.append(1 if request.form.get("director_name") == director else 0)

        for company in top_companies:
            features.append(1 if request.form.get("company_name") == company else 0)

        for country in top_countries:
            features.append(1 if request.form.get("country_name") == country else 0)

        # Convert to array and scale
        input_array = np.array(features).reshape(1, -1)
        scaled_input = scaler.transform(input_array)

        # Predict
        prediction = model.predict(scaled_input)

        return render_template("index.html", 
                               prediction_result=f"Predicted Rating Category: {prediction[0]}",
                               actors=top_actors, 
                               directors=top_directors, 
                               companies=top_companies, 
                               countries=top_countries)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
