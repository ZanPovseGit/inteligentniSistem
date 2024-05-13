import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer, StandardScaler
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
import mlflow 
import tf2onnx.convert

mlflow.set_tracking_uri("https://dagshub.com/ZanPovseGit/inteligentniSistem.mlflow")

os.environ["MLFLOW_TRACKING_USERNAME"] = "ZanPovseGit"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "bdf091cc3f58df2c8346bb8ce616545e0e40b351"


with mlflow.start_run(run_name="Grajenje modela2"):
    
    def create_lstm_model(input_shape):
        model = Sequential()
        model.add(LSTM(units=100, return_sequences=True, input_shape=input_shape))
        model.add(LSTM(units=100, return_sequences=True))
        model.add(LSTM(units=50))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
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
        
    def load_and_process_data(json_file, selected_columns):
        data = pd.read_csv(json_file)

        data_selected = data[selected_columns]

        print("Data selected columns:")
        print(data_selected.head())  

        pipeline = Pipeline([
            ('datetime_transformer', FunctionTransformer(DateTimeTransformer().transform)),
            ('scaler', MinMaxScaler())
        ])
        X_scaled = pipeline.fit_transform(data_selected)
        
        all_columns  = ['target', 'rain','day', 'month', 'year', 'hour', 'minute']

        df_lstm = pd.DataFrame(X_scaled, columns=all_columns)

        lstm_target = 'available_bike_stands'
        df_lstm[lstm_target] = data[lstm_target]

        print("Data after transformation:")
        print(df_lstm.head()) 

        y_lstm = data[lstm_target]  
        joblib.dump(pipeline, 'src/models/evaluation_scaler.pkl')

        return df_lstm, y_lstm



    learning_json_file = 'data/tempdata/raw/learning_data.csv'
    evaluation_json_file = 'data/tempdata/raw/evaluation_data.csv'

    selected_columns = ['target', 'rain', 'time']

    X_train_lstm, y_train_lstm = load_and_process_data(learning_json_file, selected_columns)
    X_eval_lstm, y_eval_lstm = load_and_process_data(evaluation_json_file, selected_columns)


    input_shape = (X_train_lstm.shape[1], 1)
    X_train_lstm = X_train_lstm.values.reshape((X_train_lstm.shape[0], X_train_lstm.shape[1], 1))
    X_eval_lstm = X_eval_lstm.values.reshape((X_eval_lstm.shape[0], X_eval_lstm.shape[1], 1))

    lstm_model = create_lstm_model(input_shape)
    lstm_model.fit(X_train_lstm, y_train_lstm, epochs=10, batch_size=32, validation_split=0.2)

    loss = lstm_model.evaluate(X_eval_lstm, y_eval_lstm)
    print(f"Test Loss: {loss}")

    y_pred = lstm_model.predict(X_eval_lstm)
    y_pred_binary = (y_pred > 0.5).astype(int)

    accuracy = accuracy_score(y_eval_lstm, y_pred_binary)
    precision = precision_score(y_eval_lstm, y_pred_binary)
    recall = recall_score(y_eval_lstm, y_pred_binary)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

    model_filename = 'src/models/evaluation_model.h5'


    lstm_model.save(model_filename)


    mlflow.tensorflow.autolog()
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision",precision)
    mlflow.log_metric("recall",recall)
    mlflow.log_param("LSTM", 100)


    try:
        previous_production_run = mlflow.search_runs(filter_string="tags.environment = 'production'",order_by=["start_time DESC"]).iloc[0]
        previous_production_run_id = previous_production_run["run_id"]
        previous_model_path = f"runs:/{previous_production_run_id}/lstm_model"
        previous_accuracy = previous_production_run["metrics.accuracy"]
        print(f"Previous model accuracy: {previous_accuracy}")
    except IndexError:
        print("No previous model found.")

    new_accuracy = accuracy_score(y_eval_lstm, y_pred_binary)
    print(f"New model accuracy: {new_accuracy}")

    if new_accuracy >= previous_accuracy:
        onnx_model, _ = tf2onnx.convert.from_keras(lstm_model)
        mlflow.onnx.log_model(onnx_model,"onnx")
        mlflow.register_model("runs:/" + mlflow.active_run().info.run_id + "/lstm_model", "GradenjeModela2")
        mlflow.log_param("environment", "production")
        print("New model saved.")
    else:
        print("New model is not better than the previous one. Keeping the old model.")




    
