from flask import Flask, jsonify, request
from mlflow.tensorflow import load_model
import joblib
import pandas as pd
import numpy as np
import mlflow

app = Flask(__name__)

class DateTimeTransformer:
        def fit(self, X, y=None):
            return self
        
        def transform(self, X, y=None):
            X['time'] = pd.to_datetime(X['time'])
            X['day'] = X['time'].dt.day
            X['month'] = X['time'].dt.month
            X['year'] = X['time'].dt.year
            X['hour'] = X['time'].dt.hour
            X['minute'] = X['time'].dt.minute
            X = X.drop('time', axis=1)
            return X

preprocessing_pipeline = joblib.load('src/models/evaluation_scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():

    previous_production_run = mlflow.search_runs(filter_string="tags.environment = 'production'",order_by=["start_time DESC"]).iloc[0]
    previous_production_run_id = previous_production_run["run_id"]
    previous_model_path = f"runs:/{previous_production_run_id}/lstm_model"
    lstm_model = load_model(previous_model_path)
    
    request_data = request.get_json()
 
    processed_data = preprocess_request_data(request_data)
    
    predictions = lstm_model.predict(processed_data)

    return jsonify({'predictions': predictions.tolist()})

def preprocess_request_data(request_data):
    data = pd.DataFrame(request_data)
    
    processed_data = preprocessing_pipeline.transform(data)
    
    processed_data = processed_data.reshape((processed_data.shape[0], processed_data.shape[1], 1))
    
    return processed_data

if __name__ == '__main__':
    app.run(debug=True)
