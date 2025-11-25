# CSC311 Fall 2025 Final Project
By: Tajwaar Shafiq,
    David Wang,
    Frederick Menses,
    and Syed Salman Husainie

## Introduction

This codebase contains all of the necessary data, code, and evidence produced as a result of our final project required by CSC311 (Introduction to Machine Learining) at the University of Toronto (UTSG).

After having all students complete a survey about different generative AI models (ChatGPT, Gemini, and Claude), the goal was to design an array of supervised learning models in order utilize NLP to classify them, and also predict the model being referenced in a set of unseen survey results.

At completion, we collectively designed 3 ways of data parsing, pre-processesing, and cleaning (`numpy`, `pandas`), alongside the implementation of the following distinct model families with thorough hyperparameter tuning (`sklearn`), including:

### 1. Decision Trees (with Boosting)
    models/decision_trees/bow_decision_tree_tuning.ipynb
    models/decision_trees/sklearn_tree_tuning.ipynb
### 2. Naive Bayes (MLE, MAP)
    models/naive_bayes/naive_bayes.ipynb
### 3. Neural Networks (MLPs)
    models/neural_networks/bow_mlp_demo.ipynb
    models/neural_networks/mlp_demo.ipynb
### 4. Random Forests
    models/decision_trees/random_forest.ipynb 
    models/decision_trees/random_forest_cmpr.ipynb
    models/decision_trees/random_forest_cmpr_std.ipynb

## Navigating the Codebase

The codebase as a whole is designed to organize the models away from the pre-requisite code. It is important to note here that all model training/tuning notebooks (located in `models/`) are NOT designed to be run, but rather serve as evidence for the work done. Also please note that any `.py` or `.ipynb` files that end with `XXX_std.py` or `XXX_std.ipynb` are designed to work without the use of the `sklearn` library.

1. `root/`

    The root directory contains given starter code files (such as `pred_example.py`, `project_baseline.py`), the `requirements.txt` for the project, alongside the initial data we were given in `training_data_clean.csv`. The final report of the project, `report.pdf` can also be found here for quick access.

2. `src/`

    The source directory encapsulates the `dataloader.py` files, used to generate the pre-processed data contained within `data/`. It also contains many helper functions (e.g. activation/gradient descent functions), classes (e.g. BoW/TF-IDF vectorizers), and constants (e.g. label mappings, MCQ/selection column indexes) within `/helpers`.
    
3. `models/`

    The models directory envelopes all the work done in training, tuning, and testing the models. They contain `.ipynb` notebooks in which this work has been captured, helpful data plots/reports, and fitted versions of the models (such that they can be loaded without the use of `sklearn`) in either `.JSON` or `.pkl` formats.

    Note that the relevant files pertaining to the Random Forest models are located within the `decision_trees/` directory. 

4. `submission/`

    The submission folder contains the final `report.pdf` and zipped and un-zipped versions of the `code/` submitted by the team. Similar to the code within the entire codebase, the `.py` and `.ipynb` files within `code/` are NOT designed to be executable alone, and may require refactoring to do so.

## Results & Report

For an overview of the project data exploration, methodology, and predictions results as a whole, we reccommend you visit the `report.pdf` file. For individual model families results can be found within the representative `models/[MODEL_NAME]/[MODEL_NOTEBOOOK].ipynb` file. 

To summarize, with our best model, we were able to achieve an impressive **98%/94%/88%** accuracy on Training/Validation/Test datasets respectively using a 500-tree Random Forest (following a BoW vectorization via `sklearn.feature_extraction.CountVectorizer`) with limited data.
