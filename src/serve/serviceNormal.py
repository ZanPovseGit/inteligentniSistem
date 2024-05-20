from flask import Flask, request, jsonify
import onnxruntime as ort
import numpy as np
import pandas as pd
import joblib
import mlflow
import os


appImp = Flask(__name__)

mlflow.set_tracking_uri("https://dagshub.com/ZanPovseGit/inteligentniSistem.mlflow")

os.environ["MLFLOW_TRACKING_USERNAME"] = "ZanPovseGit"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "bdf091cc3f58df2c8346bb8ce616545e0e40b351"


#model_path = "src/models/model.onnx/onnx/model.onnx" 
#ort_session = ort.InferenceSession(model_path)


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
    

pipeline = joblib.load('src/models/evaluation_scaler.pkl')


try:
    previous_production_run = mlflow.search_runs(filter_string="tags.environment = 'production'", order_by=["start_time DESC"]).iloc[0]
    previous_production_run_id = previous_production_run["run_id"]
    previous_model_path = f"runs:/{previous_production_run_id}/onnx"
    model_url = mlflow.artifacts.download_artifacts(run_id=previous_production_run_id, artifact_path="onnx",dst_path="src/models/model.onnx")
    ort_session = ort.InferenceSession("src/models/model.onnx/onnx/model.onnx")
    onnx_session = ort.InferenceSession("src/models/model.onnx/onnx/model.onnx")
except IndexError:
    print("No previous model found.")



@appImp.route('/predictLas', methods=['POST'])
def predict():
    data = request.get_json()

    feature_data = {
        'target': data['target'],
        'rain': data['rain'],
        'time': data['time']
    }
    
    features_df = pd.DataFrame([feature_data])

    processed_features = pipeline.transform(features_df)

    processed_features = pd.DataFrame(processed_features)
    processed_features['available_bike_stands'] = 5
    processed_features = processed_features.values

    processed_features = processed_features.reshape((processed_features.shape[0], processed_features.shape[1], 1))

    ort_inputs = {ort_session.get_inputs()[0].name: processed_features.astype(np.float32)}
    ort_outs = ort_session.run(None, ort_inputs)

    prediction = ort_outs[0]

    return jsonify({'prediction': prediction.tolist()})

def run_flask_app():
    appImp.run(debug=False, port=5000)

if __name__ == '__main__':
    run_flask_app()
