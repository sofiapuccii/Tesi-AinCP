# Time Series Analysis for the AInCP Project

This repository contains the Time Series Analysis component for the AInCP project. It is a fork incorporating advancements from the bachelor thesis work by Davide Marchi and Giordano Scerra.

## Executable Python Files

The primary executable Python scripts are located in the `Pipeline/` folder:

*   **`main_whole_assessment.py`**:
    *   Executes a 10-fold cross-validation using the entire dataset.
    *   In each fold, 90% of the data is used for training the models, and the remaining 10% is used for assessment.
    *   The results from all folds are then merged.

*   **`main_train_plot.py`**:
    *   Executes a single iteration of the analysis pipeline.
    *   Uses 10% of the data as a test set for assessment purposes.

Before running the code make sure to update the data_folder variable to indicate where the data are placed.