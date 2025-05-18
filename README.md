# Time Series Analysis for the AInCP Project

This repository contains the Time Series Analysis component for the AInCP project. It is a fork incorporating advancements from the bachelor thesis work by Davide Marchi and Giordano Scerra.

## Executable Python Files

The primary executable Python scripts are located in the `Pipeline/` folder:

*   **`main_whole_assessment.py`**: This script executes the complete 10-fold cross-validation pipeline.
    *   Executes a 10-fold cross-validation using the entire dataset.
    *   The 10 folds are processed in parallel to improve execution time.
    *   In each fold, 90% of the data is used for training the models, and the remaining 10% is used for assessment.
    *   The results from all folds are then merged.

*   **`main_train_plot.py`**: This script is primarily intended for testing and development, executing a single iteration of the analysis pipeline.
    *   Executes a single iteration of the analysis pipeline.
    *   Uses 10% of the data as a test set for assessment purposes.

## Important Note

Before running any of the scripts, ensure that the `data_folder` variable within the Python files is updated to point to the correct location of your dataset.