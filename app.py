from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and scaler
with open('knn_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Genres list
genre_list = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 'Drama', 
              'Family', 'Fantasy', 'Foreign', 'History', 'Horror', 'Music', 'Mystery', 'Romance', 
              'Science Fiction', 'TV Movie', 'Thriller', 'War', 'Western']

# Top 10 entities
top_companies = ['Paramount Pictures', 'Universal Pictures', 'Columbia Pictures',
                 'Twentieth Century Fox Film Corporation', 'New Line Cinema',
                 'Walt Disney Pictures', 'Miramax Films', 'United Artists',
                 'Village Roadshow Pictures', 'Columbia Pictures Corporation']

top_countries = ['United States of America', 'United Kingdom', 'Canada', 'Germany',
                 'France', 'Australia', 'India', 'China', 'Japan', 'Spain']

top_directors = ['Steven Spielberg', 'Woody Allen', 'Clint Eastwood', 'Martin Scorsese',
                 'Spike Lee', 'Ridley Scott', 'Robert Rodriguez', 'Renny Harlin',
                 'Steven Soderbergh', 'Oliver Stone']

top_actors = ['Leonard Nimoy', 'George Takei', 'William Shatner',
              'DeForest Kelley', 'James Doohan', 'Bruce Willis', 'Robert De Niro',
              'Samuel L. Jackson', 'Nicolas Cage', 'Johnny Depp']

@app.route('/')
def home():
    return render_template('index.html',
                           genres=genre_list,
                           companies=top_companies,
                           countries=top_countries,
                           directors=top_directors,
                           actors=top_actors)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Numeric fields
        budget = np.log1p(float(request.form['budget']))
        revenue = np.log1p(float(request.form['revenue']))
        popularity = np.log1p(float(request.form['popularity']))
        runtime = np.log1p(float(request.form['runtime']))

        # Binary flags
        is_english = 1 if request.form['is_english'] == 'Yes' else 0
        is_released = 1 if request.form['is_released'] == 'Yes' else 0
        has_top_company = 1 if request.form['has_top_company'] == 'Yes' else 0
        has_top_country = 1 if request.form['has_top_country'] == 'Yes' else 0
        is_top_director = 1 if request.form['is_top_director'] == 'Yes' else 0
        has_top_actor = 1 if request.form['has_top_actor'] == 'Yes' else 0

        # Genres
        selected_genres = request.form.getlist('genres')
        genre_flags = [1 if genre in selected_genres else 0 for genre in genre_list]

        final_input = [
            budget, revenue, popularity, runtime,
            is_english, is_released,
            has_top_company, has_top_country, is_top_director, has_top_actor
        ] + genre_flags

        column_names = ["budget", "revenue", "popularity", "runtime",
                        "is_english", "is_released",
                        "has_top_company", "has_top_country", "is_top_director", "has_top_actor"
                        ] + genre_list

        input_df = pd.DataFrame([final_input], columns=column_names)
        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)[0]
        label_map = {0: "Average", 1: "Good", 2: "Poor"}

        return render_template('index.html',
                               prediction_text=f'Movie Rating Prediction: {label_map[prediction]}',
                               genres=genre_list,
                               companies=top_companies,
                               countries=top_countries,
                               directors=top_directors,
                               actors=top_actors)

    except Exception as e:
        return render_template('index.html',
                               prediction_text=f"Error: {str(e)}",
                               genres=genre_list,
                               companies=top_companies,
                               countries=top_countries,
                               directors=top_directors,
                               actors=top_actors)

if __name__ == '__main__':
    app.run(debug=True, port=10000)
