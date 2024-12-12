from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

# Load the trained model
model_path = 'G_9_oil_category_model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

def prepare_features(input_date, sales_list):
    """
    Prepares features for the model from minimal user input.
    Args:
    - input_date (str): Date in "YYYY-MM-DD" format.
    - sales_list (list): Sales for the past 7 days [day_1_sale, ..., day_7_sale].

    Returns:
    - dict: Prepared features for the model.
    """
    # Convert input_date to datetime
    date = datetime.strptime(input_date, "%Y-%m-%d")

    # Extract basic date features
    features = {
        'Day': date.day,
        'DayOfWeek': date.weekday(),
        'month': date.month,
        'year': date.year,
        'IsWeekend': int(date.weekday() in [4, 5, 6]),
        'IsStartOfMonth': int(date.day <= 10)
    }

    # Map sales_list to previous day sales
    sales_series = pd.Series(sales_list)
    features['prev_1'] = sales_series.iloc[-1]
    features['prev_3'] = sales_series.iloc[-3]
    features['prev_7'] = sales_series.iloc[-7]
    features['avr_3'] = sales_series[-3:].mean()
    features['avr_7'] = sales_series.mean()
    features['Sales_Growth_Rate'] = (sales_series.pct_change().iloc[-1] if len(sales_list) > 1 else 0)

    return features

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prepare_features', methods=['POST'])
def prepare_features_endpoint():
    try:
        # Extract minimal input
        input_date = request.form['input_date']
        sales_list = [float(request.form[f'day_{i}_sale']) for i in range(1, 8)]

        # Prepare features
        features = prepare_features(input_date, sales_list)

        return jsonify(features)  # Return as JSON
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from form
        feature_names = [
            'Day', 'DayOfWeek', 'month', 'year', 'IsWeekend', 
            'IsStartOfMonth', 'prev_1', 'prev_3', 'prev_7', 
            'avr_3', 'avr_7', 'Sales_Growth_Rate'
        ]
        form_data = {feature: float(request.form[feature]) for feature in feature_names}

        # Create a DataFrame for the input
        input_data = pd.DataFrame([form_data])

        # Predict using the model
        prediction = model.predict(input_data)
        output = f"{prediction[0]:,.2f}"  # Format the result

        return render_template('index.html', prediction_text=f"Predicted Sales: {output}")
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
