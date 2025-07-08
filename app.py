from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form inputs
        budget = float(request.form['budget'])
        popularity = float(request.form['popularity'])
        revenue = float(request.form['revenue'])
        runtime = float(request.form['runtime'])
        vote_count = float(request.form['vote_count'])

        # Apply log transformation to match training data
        inputs = np.log1p([budget, popularity, revenue, runtime, vote_count])
        inputs = np.array(inputs).reshape(1, -1)

        # Scale the inputs
        inputs_scaled = scaler.transform(inputs)

        # Make prediction
        prediction = model.predict(inputs_scaled)[0]

        # Convert label to text
        label_map = {0: "Average", 1: "Good", 2: "Poor"}
        prediction_label = label_map.get(prediction, "Unknown")

        return render_template("index.html", prediction=prediction_label)

    except Exception as e:
        return f"Error: {e}", 400

if __name__ == '__main__':
    app.run(debug=True)
