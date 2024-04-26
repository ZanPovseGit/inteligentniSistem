import great_expectations as gx
import pandas as pd
import os
import glob

# Assuming you have already created a DataContext and an ExpectationSuite
context = gx.get_context()
expectation_suite = context.get_expectation_suite("NewSuite")

# Load your data
file_list = glob.glob("data/tempdata/processed/*TABOR.json")
if file_list:
    df = pd.read_json(file_list[0])

    # Validate your data using the ExpectationSuite
    validation_result = context.validate_dataframe(
        dataframe=df,
        expectation_suite=expectation_suite,
        batch_kwargs={"batch_identifiers": {"file_path": file_list[0]}}
    )

    # Print the validation results
    print(validation_result)
else:
    print("No files matching the pattern found.")