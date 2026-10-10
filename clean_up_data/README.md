# Clean up data steps:

Add the data cleaning jupiter notebook, we have 84 columns left. We need to handle more on missing data and which other columns to drop.

Run the code weight_loss_data_cleaned.ipynb (with the .xpt files from the source), the output is the file nhanes_84_col.csv.

Then, run the code weight_loss_data_cleaning_from_84_to_64.ipynb (with the nhanes_84_col.csv file as input), the output is the file nhanes_64_col.csv.

Then, run the code weight_loss_data_cleaning_from_64col.ipunb, input is nhanes_64_col.csv, the output is the file nhanes_cleaned.csv.

Then, run the code data_ready.ipunb, input is nhanes_cleaned.csv, the output is the file nhanes_model_ready.csv.
