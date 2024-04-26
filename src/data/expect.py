from collections.abc import Iterable
import great_expectations as gx
import pandas as pd




df = pd.read_json("data/tempdata/processed/DVORANA_TABOR.json")

expectation_suite = gx.expectations("NewSuite")
validation_results = df.validate(expectation_suite)
print(validation_results)