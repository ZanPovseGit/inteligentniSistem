from collections.abc import Iterable
import great_expectations as gx
import pandas as pd




df = pd.read_csv("data/tempdata/processed/DVORANA TABOR.json")

expectation_suite = gx.expectations("NewSuite")
validation_results = df.validate(expectation_suite)
