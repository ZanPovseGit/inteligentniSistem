import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
import os


from evidently import ColumnMapping

from evidently.report import Report
from evidently.metrics.base_metric import generate_column_metrics
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.metrics import *

from evidently.test_suite import TestSuite
from evidently.tests.base_test import generate_column_tests
from evidently.test_preset import DataStabilityTestPreset, NoTargetPerformanceTestPreset
from evidently.tests import *


df = pd.read_json('data/tempdata/processed/DVORANA_TABOR.json',skiprows=1)

df.drop(columns=['position'], inplace=True)
df.drop(columns=['banking'], inplace=True)
df.drop(columns=['bonus'], inplace=True)

df.rename(columns={'temperature_2m': 'target'}, inplace=True)
df['prediction'] = df['target'].values + np.random.normal(0, 5, df.shape[0])

print(df.head())

midpoint = len(df) // 2

reference = df.iloc[:midpoint]
current = df.iloc[midpoint:]

report = Report(metrics=[
    DataDriftPreset(),
])

report.run(reference_data=reference, current_data=current)


tests = TestSuite(tests=[
    TestNumberOfColumnsWithMissingValues(),
    TestNumberOfRowsWithMissingValues(),
    TestNumberOfConstantColumns(),
    TestNumberOfDuplicatedRows(),
    TestNumberOfDuplicatedColumns(),
    TestColumnsType(),
    TestNumberOfDriftedColumns(),
])

tests.run(reference_data=reference, current_data=current)

tests.save_html("reports/stability_test.html")

train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

train_df.to_csv('data/tempdata/raw/learning_data.csv', index=False)
test_df.to_csv('data/tempdata/raw/evaluation_data.csv', index=False)
