from flask import Flask, jsonify, request
from mlflow.tensorflow import load_model
import joblib
import pandas as pd
import numpy as np
import mlflow
import os
import urllib.request
import shutil
import onnx
import onnxruntime as ort


app = Flask(__name__)


mlflow.set_tracking_uri("https://dagshub.com/ZanPovseGit/inteligentniSistem.mlflow")

os.environ["MLFLOW_TRACKING_USERNAME"] = "ZanPovseGit"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "bdf091cc3f58df2c8346bb8ce616545e0e40b351"

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


try:
    previous_production_run = mlflow.search_runs(filter_string="tags.environment = 'production'", order_by=["start_time DESC"]).iloc[0]
    previous_production_run_id = previous_production_run["run_id"]
    previous_model_path = f"runs:/{previous_production_run_id}/onnx"
    model_url = mlflow.artifacts.download_artifacts(run_id=previous_production_run_id, artifact_path="onnx",dst_path="src/models/model.onnx")
    onnx_model = onnx.load("src/models/model.onnx/onnx/model.onnx")
    onnx_session = ort.InferenceSession("src/models/model.onnx/onnx/model.onnx")
except IndexError:
    print("No previous model found.")


@app.route('/predict', methods=['POST'])
def predict():
    
    request_data = request.get_json()
 
    processed_data = preprocess_request_data(request_data)
    
    input_name = onnx_session.get_inputs()[0].name  # Assuming single input
    output_name = onnx_session.get_outputs()[0].name  # Assuming single output
    predictions = onnx_session.run([output_name], {input_name: processed_data})[0]

    return jsonify({'predictions': predictions.tolist()})

def preprocess_request_data(request_data):

    data_list = [request_data]

    data = pd.DataFrame(data_list)
    
    processed_data = preprocessing_pipeline.transform(data)
    
    processed_data = processed_data.reshape((processed_data.shape[0], processed_data.shape[1], 1))
    
    return processed_data

if __name__ == '__main__':
    app.run(debug=True)
