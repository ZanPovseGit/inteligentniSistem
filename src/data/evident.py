import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
import os

from evidently import ColumnMapping

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import *

df = pd.read_json('data/tempdata/processed/DVORANA_TABOR.json'):

df.drop(columns=['position'], inplace=True)
df.drop(columns=['banking'], inplace=True)
df.drop(columns=['bonus'], inplace=True)

df.rename(columns={'temperature_2m': 'target'}, inplace=True)
df['prediction'] = df['target'].values + np.random.normal(0, 5, df.shape[0])

midpoint = len(df) // 2

reference = df.iloc[:midpoint]
current = df.iloc[midpoint:]

report = Report(metrics=[
    DataDriftPreset(),
])

report.run(reference_data=reference, current_data=current)
print(report.as_dict())

train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

train_df.to_csv('data/tempdata/raw/learning_data.json', index=False)
test_df.to_csv('data/tempdata/raw/evaluation_data.json', index=False)
