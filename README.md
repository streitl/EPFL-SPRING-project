# Robustness & Manipulability of Interpretable Models

## General Information
This is the repository for the MSc semester project of Luã Streit at the SPRING lab.

The repository is organised as follows:
- The folder `src` contains the python source code of the project
  - `attacks.py` contains `find_adversarial_examples`
  - `feature_selection.py` contains `forward_stepwise_regression`
  - `loader.py` contains information about the datasets, and more importantly `load_dataset` (*does not work for texas/IEEECIS yet*)
  - `models.py` contains `SRR`, the Select-Regress-Round implementation
  - `preprocessing.py` contains preprocessing methods, namely `processing_pipeline`
- The folder `data` contains the datasets' `.csv` files:
  - There are 21 UCI datasets (retrieved from the repository of the '_Simple Rules for Complex Decisions_' paper)
  - It is here where you should put the texas and IEEECIS datasets (they are too large)

There are also scripts on the main directory that allow me to verify that I didn't break anything, and also allow to reproduce some results:
- `adversarial_german.py` trains SRR on `german_credit`, and outputs the adversarial examples that were found
- `all_uci_datasets.py` trains SRR on all UCI datasets, and outputs performance metrics for each of them
- `bankruptcy_monotonicity.py` checks whether SRR trained on `bankruptcy` verifies monotonicity, for many train/test splits
- `case_study.py` imitates `case_study.R` from the repository of the '_Simple Rules for Complex Decisions_' paper

## Requirements
Non-exhaustive list of installed conda packages:
|Name                      |Version          |         Build|
| :----------------------- | :-------------: | -----------: |
|numpy                     |1.19.1           |py38hbc911f0_0|
|pandas                    |1.1.3            |py38he6710b0_0|
|pip                       |20.2.2           |        py38_0|
|python                    |3.8.5            |    h7579374_1|
|scikit-learn              |0.23.2           |py38h0573a6f_0|
|scipy                     |1.5.2            |py38h0b6359f_0|
|statsmodels               |0.11.1           |py38h7b6447c_0|
|tqdm                      |4.49.0           |          py_0|
