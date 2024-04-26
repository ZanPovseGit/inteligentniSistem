from collections.abc import Iterable
import great_expectations as gx
import pandas as pd
import os
import glob



file_list = glob.glob("data/tempdata/processed/*TABOR.json")
print(os.listdir("data/tempdata/processed/"))

if file_list:
    df = pd.read_json(file_list[0])
    expectation_suite = gx.expectations("NewSuite")
    validation_results = df.validate(expectation_suite)
    print(validation_results)
else:
    print("No files matching the pattern found.")
