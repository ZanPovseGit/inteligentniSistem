import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json
from PIL import Image
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score,recall_score
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPRegressor
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn import *

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

json_dir = 'data/tempdata/processed'




for filename in os.listdir(json_dir):
    if filename.endswith('.json'):
        with open(os.path.join(json_dir, filename), 'r') as file:
            data = json.load(file)
        df = pd.DataFrame(data)
        df = preprocess_data(df)
        lstm_features = ['temperature_2m', 'rain','day','month','year']  # features uporabljene
        lstm_target = 'available_bike_stands'
        df_lstm = df[lstm_features + [lstm_target]]
        X_lstm = df_lstm.drop(lstm_target, axis=1)
        y_lstm = df_lstm[lstm_target]
        X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)
        scaler_lstm = MinMaxScaler()
        X_train_lstm_scaled = scaler_lstm.fit_transform(X_train_lstm)
        X_test_lstm_scaled = scaler_lstm.transform(X_test_lstm)
        input_shape = (X_train_lstm_scaled.shape[1], 1)
        X_train_lstm_scaled = X_train_lstm_scaled.reshape((X_train_lstm_scaled.shape[0], X_train_lstm_scaled.shape[1], 1))
        X_test_lstm_scaled = X_test_lstm_scaled.reshape((X_test_lstm_scaled.shape[0], X_test_lstm_scaled.shape[1], 1))
        lstm_model = create_lstm_model(input_shape)
        lstm_model.fit(X_train_lstm_scaled, y_train_lstm, epochs=10, batch_size=32, validation_data=(X_test_lstm_scaled, y_test_lstm))
        loss = lstm_model.evaluate(X_test_lstm_scaled, y_test_lstm)
        print(f"Test Loss for {filename}: {loss}")
        y_pred = lstm_model.predict(X_test_lstm_scaled)
        y_pred_binary = (y_pred > 0.5).astype(int)
        print(np.unique(y_pred_binary))


        accuracy = accuracy_score(y_test_lstm, y_pred_binary)
        precision = precision_score(y_test_lstm, y_pred_binary, pos_label=0)
        recall = recall_score(y_test_lstm, y_pred_binary, pos_label=0)
        print(f"Accuracy for {filename}: {accuracy}")
        print(f"Precision for {filename}: {precision}")
        print(f"Recall for {filename}: {recall}")
        metrics_file_path = f'reports/{filename}_metrics.txt'

        with open(metrics_file_path, 'w') as metrics_file:
            metrics_file.write(f"Test Loss: {loss}\n")
            metrics_file.write(f"Accuracy: {accuracy}\n")
            metrics_file.write(f"Precision: {precision}\n")
            metrics_file.write(f"Recall: {recall}\n")

        lstm_model.save(f"src/models/{filename}.h5")

        joblib.dump(scaler_lstm, f"src/models/{filename}.pkl")
    



































#features_to_scale = ['day', 'month', 'year', 'hour', 'minute','temperature', 'relative_humidity', 'dew_point', 'apparent_temperature',
#                      'precipitation_probability', 'rain', 'surface_pressure', 'bike_stands', 'available_bike_stands']

#scaler = MinMaxScaler()
#df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

#print(df.columns)











#lstm_model.save(r'C:\Users\Uporabnik\Desktop\notebooks\inteligVaje\src\models\model.h5')

#joblib.dump(scaler_lstm, r'C:\Users\Uporabnik\Desktop\notebooks\inteligVaje\src\models\scaler.pkl')

print("Shape of X_test_lstm:", X_test_lstm.shape)

print("Input Shape of LSTM Model:", input_shape)



# Create an example input sequence for prediction
example_input_sequence = X_test_lstm[0:1]  # Take the first sequence from your test data
print("Example Input Sequence Shape:", example_input_sequence.shape)

# Make a prediction using the LSTM model
predicted_output = lstm_model.predict(example_input_sequence)
print("Predicted Output Shape:", predicted_output.shape)