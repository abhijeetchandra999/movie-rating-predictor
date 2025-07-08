@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input from form
        budget = float(request.form['budget'])
        popularity = float(request.form['popularity'])
        revenue = float(request.form['revenue'])
        runtime = float(request.form['runtime'])

        # Log transform
        input_data = np.log1p([budget, popularity, revenue, runtime])
        input_data = np.array(input_data).reshape(1, -1)

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)

        label = {0: "Average", 1: "Good", 2: "Poor"}  # adjust if needed
        result = label[prediction[0]]

        return render_template("index.html", prediction=result)
    except Exception as e:
        return f"Error: {e}"
