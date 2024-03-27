from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import joblib
import json

app = Flask(__name__)


lstm_model = load_model('models/model.h5')
scaler = joblib.load('models/scaler.pkl')

lstm_features = ['day', 'month', 'year', 'hour', 'minute', 'relative_humidity', 'dew_point', 'apparent_temperature',
                 'precipitation_probability', 'rain', 'surface_pressure', 'bike_stands', 'temperature']


@app.route('/predict', methods=['POST'])
def predict():
    try:

        data = request.get_json()

        input_data = pd.DataFrame(data, index=[0])
        input_data_scaled = scaler.transform(input_data[lstm_features])

        input_data_scaled = input_data_scaled.reshape((input_data_scaled.shape[0], input_data_scaled.shape[1], 1))

        predictions = lstm_model.predict(input_data_scaled)

        #predictions = scaler.inverse_transform(predictions)

        return jsonify({'predictions': predictions.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)})


def run_flask_app():
    app.run(debug=False, port=6666)

if __name__ == '__main__':
    run_flask_app()
