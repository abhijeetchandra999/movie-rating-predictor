from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load model and scaler
model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get numerical inputs from form
        budget = float(request.form['budget'])
        popularity = float(request.form['popularity'])
        revenue = float(request.form['revenue'])
        runtime = float(request.form['runtime'])

        # Apply log transformation (same as training)
        features = np.log1p([budget, popularity, revenue, runtime])
        features = np.array(features).reshape(1, -1)

        # Scale the input
        scaled_input = scaler.transform(features)

        # Make prediction
        prediction = model.predict(scaled_input)[0]

        # Decode prediction
        label_map = {0: "Average", 1: "Good", 2: "Poor"}
        prediction_label = label_map.get(prediction, "Unknown")

        return render_template('index.html', prediction_text=f"Predicted Rating: {prediction_label}")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
