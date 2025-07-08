from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        budget = float(request.form['budget'])
        popularity = float(request.form['popularity'])
        revenue = float(request.form['revenue'])
        runtime = float(request.form['runtime'])

        # Apply log1p transformation
        inputs = np.log1p([budget, popularity, revenue, runtime])
        inputs = np.array(inputs).reshape(1, -1)

        # Scale the inputs
        inputs_scaled = scaler.transform(inputs)

        prediction = model.predict(inputs_scaled)[0]
        label_map = {0: "Average", 1: "Good", 2: "Poor"}
        prediction_label = label_map.get(prediction, "Unknown")

        return render_template("index.html", prediction=prediction_label)
    
    except Exception as e:
        return f"Error: {e}", 400

if __name__ == '__main__':
    app.run(debug=True)
