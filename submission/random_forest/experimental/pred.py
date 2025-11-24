import sys
import numpy as np
import pandas as pd
import pickle
sys.path.append('..')

from dataloader import DataLoader
from constants import PREDICTIONS_TO_LABELS, RF_PATH, COUNT_VECTORIZER_PATH, COLUMNS_PATH

with open(COUNT_VECTORIZER_PATH, 'rb') as f:
    count_vectorizer = pickle.load(f)

def predict_all(filename):
    """
    Make predictions for the data in filename. Returns a dataframe with the predictions
    """
    
    # Extract vocab from training cols CSV
    dl = DataLoader(count_vectorizer)
    expected_columns = dl.load_column_names(COLUMNS_PATH)

    # Process test data with expected columns
    preprocessed_data = dl.preprocess(filename, expected_columns)
    
    # Generate model from saved .pkl
    with open(RF_PATH, 'rb') as f:
        rf = pickle.load(f)
    
    # Make and clean predictions
    base_predictions = rf.predict(preprocessed_data.to_numpy())
    vectorized_map = np.vectorize(PREDICTIONS_TO_LABELS.get)
    return vectorized_map(base_predictions)
    