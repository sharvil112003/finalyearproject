from flask import Flask, request, render_template
import os
import h5py
import pickle
import numpy as np
import datetime
import yfinance as yf
from tensorflow.keras.models import model_from_json

# Initialize Flask app
app = Flask(__name__)

# Load models from .h5 file
def load_models(models_path, nifty50_companies):
    models = {}
    with h5py.File(models_path, "r") as h5_file:
        for company in nifty50_companies:
            try:
                model_group = h5_file[company]
                model_json = model_group.attrs["model_json"]
                model = model_from_json(model_json)

                # Dynamically construct the weights path
                weights_path = os.path.join("weights", f"{company}_weights.weights.h5")
                if not os.path.exists(weights_path):
                    raise FileNotFoundError(f"Weights file not found: {weights_path}")

                # Load weights
                model.load_weights(weights_path)
                models[company] = model
            except Exception as e:
                print(f"Error loading model for {company}: {e}")
                continue
    return models

# Load scalers from .pkl file
def load_scalers(scalers_path):
    try:
        with open(scalers_path, "rb") as f:
            scalers = pickle.load(f)
        return scalers
    except Exception as e:
        raise RuntimeError(f"Error loading scalers from {scalers_path}: {e}")

# Function to predict high and low prices
def predict(stock, days_ahead, models, scalers):
    # Download stock data for the last 'prediction_days'
    try:
        data = yf.download(stock, start="2010-01-01", end=datetime.datetime.now().strftime("%Y-%m-%d"))
        data.dropna(inplace=True)

        # Get the latest 60 days of data for prediction
        prediction_days = 60
        last_60_days = data[['High', 'Low']].tail(prediction_days)

        # Scale the data
        scaler = scalers.get(stock)
        if not scaler:
            raise ValueError(f"No scaler found for {stock}")

        scaled_data = scaler.transform(last_60_days.values)

        # Prepare input for the model
        x_test = [scaled_data]
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 2))  # Reshape for LSTM input

        # Predict for the requested days ahead
        model = models.get(stock)
        if not model:
            raise ValueError(f"No model found for {stock}")

        predicted_values = []
        for _ in range(days_ahead):
            predicted = model.predict(x_test)
            predicted_values.append(predicted)
            new_input = np.append(x_test[0][1:], predicted, axis=0)
            x_test = np.reshape([new_input], (1, prediction_days, 2))

        # Convert predictions back to original scale
        predicted_values = np.array(predicted_values).reshape(-1, 2)
        predicted_values = scaler.inverse_transform(predicted_values)

        return predicted_values
    except Exception as e:
        raise RuntimeError(f"Error during prediction: {e}")

# Load models and scalers globally
MODELS_PATH = os.path.join("models", "all_models.h5")
SCALERS_PATH = os.path.join("scalers", "all_scalers.pkl")
NIFTY50_COMPANIES = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 
                     'HINDUNILVR.NS', 'BAJFINANCE.NS', 'SBIN.NS', 'KOTAKBANK.NS', 'BHARTIARTL.NS']

try:
    models = load_models(MODELS_PATH, NIFTY50_COMPANIES)
    scalers = load_scalers(SCALERS_PATH)
except Exception as e:
    print(f"Error during global initialization: {e}")
    models = {}
    scalers = {}

import matplotlib
import matplotlib.pyplot as plt
from io import BytesIO
import base64
matplotlib.use('Agg')

def generate_combined_graph(historical_data, predicted_values, days_ahead):
    # Ensure you access the Series and convert it to a list
    high_prices_actual = historical_data['High'].tail(60).values.tolist()  # Last 60 days
    low_prices_actual = historical_data['Low'].tail(60).values.tolist()

    # Extract predicted values
    high_prices_predicted = [p[0] for p in predicted_values]
    low_prices_predicted = [p[1] for p in predicted_values]

    # Define days for plotting
    historical_days = list(range(1, len(high_prices_actual) + 1))
    prediction_days = list(range(len(high_prices_actual) + 1, len(high_prices_actual) + days_ahead + 1))

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(historical_days, high_prices_actual, label='Historical High', color='blue', linestyle='--')
    plt.plot(historical_days, low_prices_actual, label='Historical Low', color='orange', linestyle='--')
    plt.plot(prediction_days, high_prices_predicted, label='Predicted High', color='blue', marker='o')
    plt.plot(prediction_days, low_prices_predicted, label='Predicted Low', color='orange', marker='o')
    plt.fill_between(prediction_days, high_prices_predicted, low_prices_predicted, color='lightblue', alpha=0.3)
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.title('Historical and Predicted Prices')
    plt.legend()
    plt.grid()
    plt.tight_layout()

    # Convert plot to Base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    graph_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()
    plt.close()

    return graph_base64



def generate_sentiment_graph(titles, sentiments):
    # Plot the sentiment values
    plt.figure(figsize=(10, 6))
    colors = ['green' if s > 0 else 'red' for s in sentiments]
    plt.bar(titles, sentiments, color=colors)
    plt.xticks(rotation=90)
    plt.ylabel('Sentiment Score')
    plt.title('Sentiment Analysis Results')
    plt.tight_layout()

    # Save the plot to a BytesIO buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # Convert to Base64 string
    graph_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()
    plt.close()

    return graph_base64


@app.route('/')
def index():
    return render_template('index.html', companies=NIFTY50_COMPANIES)

@app.route('/predict', methods=['POST'])
def predict_route():
    stock = request.form['stock']
    days_ahead = int(request.form['days_ahead'])

    if stock not in NIFTY50_COMPANIES:
        return render_template('result.html', error_message=f"Stock '{stock}' not recognized.")

    if days_ahead not in [1, 7, 30, 365]:
        return render_template('result.html', error_message="Invalid number of days ahead. Select 1, 7, 30, or 365.")

    try:
        predictions = predict(stock, days_ahead, models, scalers)

        # Example sentiment analysis data
        titles = ["Stock is rising", "Market seems stable", "Volatility expected"]
        sentiments = [0.8, 0.5, -0.3]

        # Generate graphs
        sentiment_graph = generate_sentiment_graph(titles, sentiments)

        # Example historical data (replace with actual)
        data = yf.download(stock, start="2022-01-01", end=datetime.datetime.now().strftime("%Y-%m-%d"))
        combined_graph = generate_combined_graph(data, predictions, days_ahead)

        return render_template(
            'result.html',
            stock=stock,
            days_ahead=days_ahead,
            prediction=predictions[-1],  # High and Low for the last day predicted
            titles=titles,
            sentiments=sentiments,
            sentiment_graph=sentiment_graph,
            combined_graph=combined_graph,  # Added combined graph
            zip=zip,
            error_message=None
        )
    except Exception as e:
        return render_template('result.html', error_message=str(e))

if __name__ == '__main__':
    app.run(debug=True)
