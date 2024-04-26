import pandas as pd
import numpy as np

from sklearn.datasets import fetch_california_housing

from evidently import ColumnMapping

from evidently.report import Report
from evidently.metrics.base_metric import generate_column_metrics
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.metrics import *

from evidently.test_suite import TestSuite
from evidently.tests.base_test import generate_column_tests
from evidently.test_preset import DataStabilityTestPreset, NoTargetPerformanceTestPreset
from evidently.tests import *

df = pd.read_json('data/tempdata/processed/DVORANA_TABOR.json')

df.drop(columns=['position'], inplace=True)

df.rename(columns={'temperature_2m': 'target'}, inplace=True)
df['prediction'] = df['target'].values + np.random.normal(0, 5, df.shape[0])

midpoint = len(df) // 2

reference = df.iloc[:midpoint]
current = df.iloc[midpoint:]


report = Report(metrics=[
    DataDriftPreset(), 
])

report.run(reference_data=reference, current_data=current)
report