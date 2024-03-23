from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import joblib
import glob
import json
import os
import requests

app = Flask(__name__)

def fetch_json_from_api(api_url):
    try:
        response = requests.get(api_url)
        response.raise_for_status()  
        json_data = response.json()
        return json_data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from API: {e}")
        return None
    
def format_time(df):

        df['time'] = pd.to_datetime(df['time'])

        df['day'] = df['time'].dt.day
        df['month'] = df['time'].dt.month
        df['year'] = df['time'].dt.year
        df['hour'] = df['time'].dt.hour

        df = df.drop('time', axis=1)
        return df


model_dir = r'C:\Users\Uporabnik\Desktop\notebooks\InteligVaje\src\models'


lstm_features = ['temperature_2m', 'rain','day','month','year','hour'] 

@app.route('/predict', methods=['GET'])
def predict():
    try:

        api_url2 = 'https://api.open-meteo.com/v1/forecast?latitude=46.5547&longitude=15.6467&current=temperature_2m,rain,weather_code&timezone=Europe%2FBerlin&forecast_days=1'
        weather_data = fetch_json_from_api(api_url2)
        dataWetaher = request.get_json()
        temperature_2m = dataWetaher['daily']['temperature_2m_max']
        code = dataWetaher['daily']['weather_code']
        time = dataWetaher['daily']['time']
        rain = dataWetaher['daily']['rain']

        predictions_all_intervals = []

        for filename in os.listdir(model_dir):
            if filename.endswith('.h5'):
                base_name = os.path.splitext(filename)[0]
                scaler = glob.glob(f"{model_dir}\\{base_name}.*")
                h5_model_path = os.path.join(model_dir, filename)
                h5_model = load_model(h5_model_path)        
                scaler_model_path = os.path.join(model_dir, filename)
                scaler_model = joblib.load(scaler_model_path)

                predict = []
                for i in range(len(time)):
                    # Create a dataset for each day
                    dataset = {
                        'time': time[i],
                        'weather_code': code[i],
                        'temperature_2m_max': temperature_2m[i],
                        'rain': rain[i]
                    }
                    df = pd.DataFrame(dataset)
                    df = format_time(df)
                    input_data_scaled = scaler.transform(df[lstm_features])
                    input_data_scaled = input_data_scaled.reshape((input_data_scaled.shape[0], input_data_scaled.shape[1], 1))

                    prediction = h5_model.predict(input_data_scaled)

                    predict.append(prediction.tolist())

                predictions_all_intervals.append(predict)

        return predictions_all_intervals



    except Exception as e:
        return jsonify({'error': str(e)})





def run_flask_app():
    app.run(debug=False, port=6666)

if __name__ == '__main__':
    run_flask_app()
