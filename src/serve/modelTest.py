import onnxruntime as ort
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import mlflow
import os
import joblib


mlflow.set_tracking_uri("https://dagshub.com/ZanPovseGit/inteligentniSistem.mlflow")

os.environ["MLFLOW_TRACKING_USERNAME"] = "ZanPovseGit"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "bdf091cc3f58df2c8346bb8ce616545e0e40b351"



with mlflow.start_run(run_name="DailyCheck"):

    class DateTimeTransformer:
        def fit(self, X, y=None):
            return self
            
        def transform(self, X, y=None):
            X_copy = X.copy() 
            X_copy['time'] = pd.to_datetime(X_copy['time'])
            X_copy['day'] = X_copy['time'].dt.day
            X_copy['month'] = X_copy['time'].dt.month
            X_copy['year'] = X_copy['time'].dt.year
            X_copy['hour'] = X_copy['time'].dt.hour
            X_copy['minute'] = X_copy['time'].dt.minute
            X_copy = X_copy.drop('time', axis=1)
            return X_copy
        
    def load_and_process_data(json_file, selected_columns):
        data = pd.read_csv(json_file)

        data_selected = data[selected_columns]
        
        pipeline = joblib.load('src/models/evaluation_scaler.pkl')

        X_scaled = pipeline.fit_transform(data_selected)
            
        all_columns  = ['target', 'rain','day', 'month', 'year', 'hour', 'minute']

        df_lstm = pd.DataFrame(X_scaled, columns=all_columns)

        lstm_target = 'available_bike_stands'
        df_lstm[lstm_target] = data[lstm_target]

        y_lstm = data[lstm_target]  

        return df_lstm, y_lstm


    try:
        previous_production_run = mlflow.search_runs(filter_string="tags.environment = 'production'", order_by=["start_time DESC"]).iloc[0]
        previous_production_run_id = previous_production_run["run_id"]
        previous_model_path = f"runs:/{previous_production_run_id}/onnx"
        model_url = mlflow.artifacts.download_artifacts(run_id=previous_production_run_id, artifact_path="onnx",dst_path="src/models/model.onnx")
        ort_session = ort.InferenceSession("src/models/model.onnx/onnx/model.onnx")
        onnx_session = ort.InferenceSession("src/models/model.onnx/onnx/model.onnx")
    except IndexError:
        print("No previous model found.")

    selected_columns = ['target', 'rain', 'time']

    test_data_path = 'data/tempdata/raw/evaluation_data.csv' 
    X_test_lstm, y_test_lstm = load_and_process_data(test_data_path , selected_columns)


    input_shape = (X_test_lstm.shape[1], 1)
    X_test_lstm = X_test_lstm.values.reshape((X_test_lstm.shape[0], X_test_lstm.shape[1], 1))


    model_path = "src/models/model.onnx/onnx/model.onnx"  
    ort_session = ort.InferenceSession(model_path)

    ort_inputs = {ort_session.get_inputs()[0].name: X_test_lstm.astype(np.float32)}
    y_pred = ort_session.run(None, ort_inputs)[0]

    mse = mean_squared_error(y_test_lstm, y_pred)
    mae = mean_absolute_error(y_test_lstm, y_pred)

    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")

    mlflow.log_param("mse",mse)
    mlflow.log_param("mae",mae)
