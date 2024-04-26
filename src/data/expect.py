import great_expectations as gx
import pandas as pd
import os
import glob

context = DataContext('gx')

expectation_suite = context.get_expectation_suite("NewSuite")


# Load your data
file_list = glob.glob("data/tempdata/processed/*TABOR.json")
if file_list:
    json_data = pd.read_json(file_list[0])
    validation_results = context.run_validation_operator(
    "action_list_operator",  # You might need to adjust this based on your setup
    assets_to_validate=[json_data],  # Pass the JSON data to be validated
    run_id="id1"  # Provide a unique run_id for tracking
    )

    # Check validation results
    if validation_results["success"]:
        print("Data meets expectation!")
    else:
        print("Data does not meet expectation. Validation errors:")
        for result in validation_results["validation_result"]:
            print(result)

else:
    print("No files matching the pattern found.")