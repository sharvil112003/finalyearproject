from flask import Flask, request, render_template, jsonify
import os
import h5py
import pickle
import numpy as np
import datetime
import yfinance as yf
from tensorflow.keras.models import model_from_json
import matplotlib
import matplotlib.pyplot as plt
from io import BytesIO
import base64

VALID_PERIODS = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']

# Configure matplotlib for non-interactive use
matplotlib.use('Agg')

# Initialize Flask app
app = Flask(__name__)

# Load models from .h5 file
def load_models(models_path, nifty50_companies):
    models = {}
    try:
        with h5py.File(models_path, "r") as h5_file:
            for company in nifty50_companies:
                try:
                    model_group = h5_file[company]
                    model_json = model_group.attrs["model_json"]
                    model = model_from_json(model_json)

                    # Load weights
                    weights_path = os.path.join("weights", f"{company}_weights.weights.h5")
                    if not os.path.exists(weights_path):
                        raise FileNotFoundError(f"Weights file not found: {weights_path}")
                    model.load_weights(weights_path)
                    models[company] = model
                except Exception as e:
                    print(f"Error loading model for {company}: {e}")
    except Exception as e:
        print(f"Error loading models: {e}")
    return models

# Load scalers from .pkl file
def load_scalers(scalers_path):
    try:
        with open(scalers_path, "rb") as f:
            scalers = pickle.load(f)
        return scalers
    except Exception as e:
        raise RuntimeError(f"Error loading scalers: {e}")

# Predict high and low prices
def predict(stock, days_ahead, models, scalers):
    try:
        # Try fetching data with .NS suffix first
        stock_ticker = stock
        data = yf.download(stock_ticker, start="2010-01-01", end=datetime.datetime.now().strftime("%Y-%m-%d"))

        # If no data is found, retry without .NS
        if data.empty:
            stock_ticker = stock.replace('.NS', '')
            data = yf.download(stock_ticker, start="2010-01-01", end=datetime.datetime.now().strftime("%Y-%m-%d"))

        if data.empty:
            raise ValueError(f"No data found for {stock}")

        data.dropna(inplace=True)

        # Get the latest 60 days of data for prediction
        prediction_days = 60
        last_60_days = data[['High', 'Low']].tail(prediction_days)

        # Scale the data
        scaler = scalers.get(stock)
        if not scaler:
            raise ValueError(f"No scaler found for {stock}")

        scaled_data = scaler.transform(last_60_days.values)
        x_test = np.array([scaled_data]).reshape(1, prediction_days, 2)

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

        predicted_values = np.array(predicted_values).reshape(-1, 2)
        return scaler.inverse_transform(predicted_values)
    except Exception as e:
        raise RuntimeError(f"Error during prediction: {e}")

# Generate a combined historical and predicted graph
def generate_combined_graph(historical_data, predicted_values, days_ahead):
    try:
        high_prices_actual = historical_data['High'].tail(60).values.tolist()
        low_prices_actual = historical_data['Low'].tail(60).values.tolist()
        high_prices_predicted = [p[0] for p in predicted_values]
        low_prices_predicted = [p[1] for p in predicted_values]

        historical_days = list(range(1, len(high_prices_actual) + 1))
        prediction_days = list(range(len(high_prices_actual) + 1, len(high_prices_actual) + days_ahead + 1))

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

        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        graph_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        buffer.close()
        plt.close()

        return graph_base64
    except Exception as e:
        raise RuntimeError(f"Error generating graph: {e}")

# Generate sentiment analysis bar graph
def generate_sentiment_graph(titles, sentiments):
    try:
        plt.figure(figsize=(10, 6))
        colors = ['green' if s > 0 else 'red' for s in sentiments]
        plt.bar(titles, sentiments, color=colors)
        plt.xticks(rotation=90)
        plt.ylabel('Sentiment Score')
        plt.title('Sentiment Analysis Results')
        plt.tight_layout()

        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        graph_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        buffer.close()
        plt.close()

        return graph_base64
    except Exception as e:
        raise RuntimeError(f"Error generating sentiment graph: {e}")

# API for dynamic chart data fetching
@app.route('/get_chart_data', methods=['POST'])
def get_chart_data():
    try:
        stock = request.json.get('stock')
        period = request.json.get('period')

        if not stock or not period:
            return jsonify({"error": "Stock and period are required."}), 400

        if period not in VALID_PERIODS:
            return jsonify({"error": f"Invalid period '{period}'. Must be one of {VALID_PERIODS}."}), 400

        # Attempt to fetch data with '.NS'
        stock_ticker = stock
        interval = "1h" if period == "1d" else "1d"
        data = yf.download(tickers=stock_ticker, period=period, interval=interval)

        # Retry without '.NS' if no data is found
        if data.empty:
            stock_ticker = stock.replace('.NS', '')
            data = yf.download(tickers=stock_ticker, period=period, interval=interval)

        # If still no data, return error
        if data.empty:
            return jsonify({"error": f"No data available for stock '{stock}'. This stock may be delisted or inactive."}), 400

        # Ensure the 'Close' column exists
        if 'Close' not in data.columns:
            return jsonify({"error": f"'Close' column missing in data for stock '{stock}'."}), 400

        # Convert index and 'Close' column to lists
        labels = data.index.strftime('%Y-%m-%d %H:%M').tolist()
        prices = data['Close'].tolist()

        return jsonify({"labels": labels, "prices": prices}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Homepage route
@app.route('/')
def index():
    valid_companies = []
    for company in NIFTY50_COMPANIES:
        data = yf.download(company, period="1d", interval="1d")
        if not data.empty:
            valid_companies.append(company)

    return render_template('index.html', companies=valid_companies)


# Predict route
@app.route('/predict', methods=['POST'])
def predict_route():
    stock = request.form['stock']
    days_ahead = int(request.form['days_ahead'])

    if stock not in NIFTY50_COMPANIES:
        return render_template('result.html', error_message=f"Stock '{stock}' not recognized.")

    try:
        predictions = predict(stock, days_ahead, models, scalers)
        titles = ["Stock is rising", "Market seems stable", "Volatility expected"]
        sentiments = [0.8, 0.5, -0.3]

        sentiment_graph = generate_sentiment_graph(titles, sentiments)
        data = yf.download(stock, start="2022-01-01", end=datetime.datetime.now().strftime("%Y-%m-%d"))

        if data.empty:
            return render_template(
                'result.html',
                error_message=f"No historical data available for stock '{stock}'."
            )

        combined_graph = generate_combined_graph(data, predictions, days_ahead)

        # Pass `zip` explicitly
        return render_template(
            'result.html',
            stock=stock,
            days_ahead=days_ahead,
            prediction=predictions[-1],
            titles=titles,
            sentiments=sentiments,
            sentiment_graph=sentiment_graph,
            combined_graph=combined_graph,
            zip=zip,  # Pass the `zip` function explicitly
            error_message=None
        )
    except Exception as e:
        return render_template('result.html', error_message=str(e))

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

if __name__ == '__main__':
    app.run(debug=True)
