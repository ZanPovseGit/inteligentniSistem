import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
import json
import mlflow 

def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def preprocess_data(df):
    df['time'] = pd.to_datetime(df['time'])
    df['day'] = df['time'].dt.day
    df['month'] = df['time'].dt.month
    df['year'] = df['time'].dt.year
    df['hour'] = df['time'].dt.hour
    df['minute'] = df['time'].dt.minute
    df = df.drop('time', axis=1)
    return df

def load_and_process_data(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)

    df = pd.DataFrame(data)
    df = preprocess_data(df)

    lstm_features = ['temperature_2m', 'rain','day','month','year']
    lstm_target = 'available_bike_stands'

    df_lstm = df[lstm_features + [lstm_target]]
    X_lstm = df_lstm.drop(lstm_target, axis=1)
    y_lstm = df_lstm[lstm_target]

    return X_lstm, y_lstm

learning_json_file = 'data/tempdata/raw/learning_data.json'
evaluation_json_file = 'data/tempdata/raw/evaluation_data.json'

X_train_lstm, y_train_lstm = load_and_process_data(learning_json_file)

X_eval_lstm, y_eval_lstm = load_and_process_data(evaluation_json_file)

scaler_lstm = MinMaxScaler()
X_train_lstm_scaled = scaler_lstm.fit_transform(X_train_lstm)
X_eval_lstm_scaled = scaler_lstm.transform(X_eval_lstm)

input_shape = (X_train_lstm_scaled.shape[1], 1)
X_train_lstm_scaled = X_train_lstm_scaled.reshape((X_train_lstm_scaled.shape[0], X_train_lstm_scaled.shape[1], 1))
X_eval_lstm_scaled = X_eval_lstm_scaled.reshape((X_eval_lstm_scaled.shape[0], X_eval_lstm_scaled.shape[1], 1))

lstm_model = create_lstm_model(input_shape)
lstm_model.fit(X_train_lstm_scaled, y_train_lstm, epochs=10, batch_size=32, validation_split=0.2)

loss = lstm_model.evaluate(X_eval_lstm_scaled, y_eval_lstm)
print(f"Test Loss: {loss}")

y_pred = lstm_model.predict(X_eval_lstm_scaled)
y_pred_binary = (y_pred > 0.5).astype(int)

accuracy = accuracy_score(y_eval_lstm, y_pred_binary)
precision = precision_score(y_eval_lstm, y_pred_binary, pos_label=0)
recall = recall_score(y_eval_lstm, y_pred_binary, pos_label=0)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")

model_filename = 'src/models/evaluation_model.h5'
scaler_filename = 'src/models/evaluation_scaler.pkl'


mlflow.tensorflow.autolog()
mlflow.log_metric("accuracy", accuracy)
mlflow.set_tracking_uri("https://dagshub.com/ZanPovseGit/inteligentniSistem.mlflow")


lstm_model.save(model_filename)
joblib.dump(scaler_lstm, scaler_filename)
