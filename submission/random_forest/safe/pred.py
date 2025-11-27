import numpy as np
import sys
import pickle
sys.path.append('..')

from constants import PREDICTIONS_TO_LABELS, RF_PATH, VOCAB_PATH, COLUMNS_PATH, UNNAMED
from random_forest import RandomForestModel
from dataloader import DataLoader

def predict_all(filename):
    """
    Make predictions for the data in filename. Returns a dataframe with the predictions
    """
    
    # Extract vocab from training cols CSV
    dl = DataLoader()
    dl.extract_vocab_from_columns_csv(COLUMNS_PATH, VOCAB_PATH)
    
    # Create new dataloader and load vocab and expected columns 
    dataloader = DataLoader()
    dataloader.load_vocab(VOCAB_PATH)
    expected_cols = dataloader.load_column_names(COLUMNS_PATH)
    
    # Process test data with expected columns
    preprocessed_data = dataloader.preprocess(filename, expected_cols)
    preprocessed_data.insert(1, 'Unnamed: 0', preprocessed_data.index)
    
    # Generate model from saved JSON
    rf = RandomForestModel(RF_PATH)
    
    # Make and clean predictions
    base_predictions = rf.predict(preprocessed_data.to_numpy())
    vectorized_map = np.vectorize(PREDICTIONS_TO_LABELS.get)
    return vectorized_map(base_predictions)
    