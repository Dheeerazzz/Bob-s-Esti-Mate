from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load(r"XGB.pkl")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        sqft = float(request.form['sqft'])
        floors = int(request.form['floors'])
        print(sqft)
        # Make a prediction using your machine learning model
        prediction_scientific = model.predict([[sqft, floors]])

        print(prediction_scientific)
        prediction_values = [float(val) for val in prediction_scientific[0]]
        print(prediction_values)
        return render_template('results.html', prediction=prediction_values)

if __name__ == '__main__':
    app.run(debug=True)
