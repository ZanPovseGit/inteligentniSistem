import great_expectations as gx
from great_expectations import DataContext, DataAsset
import pandas as pd
import os
import glob

context = DataContext('gx')

expectation_suite = context.get_expectation_suite("NewSuite")


# Load your data
file_list = glob.glob("data/tempdata/processed/*TABOR.json")
if file_list:
    json_data = pd.read_json(file_list[0])
    data_asset = DataAsset(json_data)

    # Run validation
    validation_results = context.run_validation_operator(
        "default_validation_operator",  # Use appropriate validation operator
        assets_to_validate=[data_asset],  # Pass the DataAsset to be validated
        run_id="some_run_id"  # Provide a unique run_id for tracking
    )
    if validation_results["success"]:
        print("Data meets expectation!")
    else:
        print("Data does not meet expectation. Validation errors:")
        for result in validation_results["validation_result"]:
            print(result)

else:
    print("No files matching the pattern found.")