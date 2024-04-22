from collections.abc import Iterable
import great_expectations as gx
import pandas as pd




df = pd.read_csv("data/tempdata/DVORANA TABOR.csv")

expectation_suite = gx.expectations("NewSuite")
validation_results = df.validate(expectation_suite)






#context = gx.get_context()




#expectation_suite = gx.expectations("NewSuite")

#validator = context.get_validator(
#    expectation_suite_name="NewSuite",
#)

#validator.expect_column_values_to_not_be_null("name")
#expectation_suite.expect_column_values_to_be_unique("address")
#expectation_suite.expect_column_values_to_be_between("temperature_2m", min_value=0, max_value=40)

#validator.save_expectation_suite()

# Create a checkpoint
#checkpoint = context.add_or_update_checkpoint(
#    name="NewCheckpoint",
#    validator=validator,
#)

# Run the checkpoint
#checkpoint_result = checkpoint.run()

# Print the result
#print(checkpoint_result)


#expectation_suite.save()

#validator.head()

#NewSuite